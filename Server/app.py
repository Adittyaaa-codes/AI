import os
import sys
import re
import tempfile
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt
from pydantic import BaseModel
from qdrant_client import models
import os, sys, re, tempfile, uuid, uvicorn
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import os, sys, re, tempfile, uuid
from fastapi import Header,Depends, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from nearest .env file (searches upward)
load_dotenv(find_dotenv())

from Agents.multi_agent import rag_app_ex, rag_app_qa
from Utils.utility import get_user_id, collection_name_for, embedding_model, active_user_id

def clean_text(text: str) -> str:
    """Clean PDF text by removing extra whitespace and newlines"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

app = FastAPI(
    title="RAG Multi-Agent API",
    description="AI-powered RAG system with multi-agent routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str | None = None
    messages: list | None = None
    
class UploadResponse(BaseModel):
    message: str
    collection: str
    processed: int
    failed: int
    total_chunks: int
    details: dict
    
class IndexTextRequest(BaseModel):
    source: str
    text: str
    doc_id: str | None = None

    

# =====================
# Auth: Bearer JWT
# =====================
auth_scheme = HTTPBearer(auto_error=True)

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)) -> str:
    """Validate JWT from Authorization: Bearer <token> and return user identifier.

    Supports common claim keys: 'user_id', '_id', 'sub', or 'id'.
    """
    token = credentials.credentials
    try:
        secret = os.getenv("JWT_SECRET") or os.getenv("ACCESS_SECRET_KEY")
        if not secret:
            raise HTTPException(status_code=500, detail="JWT secret not configured")
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
        )
        user_id = (
            payload.get("user_id")
            or payload.get("_id")
            or payload.get("sub")
            or payload.get("id")
        )
        if not user_id:
            raise HTTPException(status_code=401, detail="User id not found in token")
        return str(user_id)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/")
async def root():
    return {
        "message": "RAG Multi-Agent API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

def _load_file_to_docs(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(path)
        docs = loader.load()
    else:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)

def _unique_save_path(base_dir: str, filename: str) -> str:
    name, ext = os.path.splitext(filename)
    candidate = os.path.join(base_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{name} ({i}){ext}")
        i += 1
    return candidate

@app.post("/upload_docs", response_model=UploadResponse)
async def upload_docs(
    files: List[UploadFile] = File(...), 
    subject: str = Form(...),
    chapter: str = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user_id: str = Depends(verify_jwt)
):
    active_user_id.set(user_id)
    processed: int = 0
    failed: int = 0
    details: dict = {}
    tmp_paths = []
    base_dir = os.path.join(os.path.dirname(__file__), "uploads", user_id)
    os.makedirs(base_dir, exist_ok=True)
    for f in files:
        try:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in [".pdf", ".txt", ".md", ".docx", ".doc"]:
                failed += 1
                details[f.filename] = "unsupported"
                continue
            save_path = _unique_save_path(base_dir, os.path.basename(f.filename))
            data = await f.read()
            with open(save_path, "wb") as out:
                out.write(data)
            tmp_paths.append((f.filename, save_path))
        except Exception as e:
            failed += 1
            details[f.filename] = str(e)
    all_docs = []
    for orig, path in tmp_paths:
        try:
            docs = _load_file_to_docs(path)
            if not docs:
                failed += 1
                details[orig] = "loaded empty docs"
                continue
                
            for d in docs:
                if d.metadata is None:
                    d.metadata = {}
                d.metadata["user_id"] = user_id
                d.metadata["subject"] = subject
                d.metadata["chapter"] = chapter or ''
                d.metadata["source"] = orig
                d.metadata["doc_id"] = str(uuid.uuid4())
                d.metadata["file_path"] = path
            all_docs.extend(docs)
            processed += 1
        except Exception as e:
            failed += 1
            details[orig] = f"processing error: {str(e)}"
        finally:
            # Clean up temp file if needed, but we might want to keep it if indexing fails?
            # For now we'll rely on the backend or a cleanup task.
            pass
    if all_docs:
        def index_in_background(docs, uid):
            try:
                QdrantVectorStore.from_documents(
                    documents=docs,
                    embedding=embedding_model,
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                    collection_name=collection_name_for(uid),
                )
                print(f"Successfully indexed {len(docs)} chunks for user {uid}")
            except Exception as e:
                print(f"Error in background indexing for user {uid}: {str(e)}")

        background_tasks.add_task(index_in_background, all_docs, user_id)
    return UploadResponse(
        message="ok",
        collection=collection_name_for(user_id),
        processed=processed,
        failed=failed,
        total_chunks=len(all_docs),
        details=details,
    )

    
@app.get("/list_docs")
async def list_documents(user_id: str = Depends(verify_jwt)):
    """List all unique source documents in the collection"""
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        
        records = client.scroll(
            collection_name=collection_name_for(user_id),
            limit=1000,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
        )
        
        sources = set()
        for record in records[0]:
            if record.payload and 'metadata' in record.payload:
                source = record.payload['metadata'].get('source')
                if source:
                    sources.add(source)
        
        return {
            "documents": list(sources),
            "count": len(sources)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_docs/{filename}")
async def delete_document(
    filename: str,
    user_id: str = Depends(verify_jwt)
):
    """Delete a single document by filename for a specific user"""
    
    try:
        from qdrant_client import QdrantClient, models
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        collection_name = collection_name_for(user_id)
        
        try:
            client.get_collection(collection_name)
        except:
            raise HTTPException(
                status_code=404, 
                detail=f"Collection not found for user {user_id}"
            )
        
        result = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filename)
                        ),
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        ),
                    ]
                )
            )
        )
        
        if result.status == models.UpdateStatus.COMPLETED:
            return {
                "success": True,
                "filename": filename,
                "collection": collection_name,
                "message": f"Document '{filename}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Deletion operation did not complete"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _extract_query(payload: dict) -> str | None:
    msgs = (payload or {}).get("messages") or []
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        parts = m.get("parts")
        if isinstance(parts, list):
            texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
            if texts:
                return "".join(texts).strip()
        content = m.get("content")
        if isinstance(content, list):
            texts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
            if texts:
                return "".join(texts).strip()
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(m.get("text"), str) and m["text"].strip():
            return m["text"].strip()
    return None

@app.post("/chat/qa")
async def stream_response(req: Request, user_id: str = Depends(verify_jwt)):
    active_user_id.set(user_id)
    body = await req.json()
    query = (body or {}).get("query") or _extract_query(body)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    async def generate():
        try:
            async for event in rag_app_qa.astream_events(
            {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "user_id": user_id,
            },
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
        
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
    
@app.post("/chat/explain")
async def stream_response(request: Request, user_id: str = Depends(verify_jwt)):
    active_user_id.set(user_id)
    body = await request.json()
    query = (body or {}).get("query") or _extract_query(body)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    async def generate():
        try:
            async for event in rag_app_ex.astream_events(
                {"messages": [HumanMessage(content=query)], "user_id": user_id, "query": query},
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
        
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
   
from typing import Optional 
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
    
class SummaryRequest(BaseModel):
    youtube_url: str
    summary_length: Optional[str] = "short"  # "short", "medium", "detailed"

class SummaryResponse(BaseModel):
    video_title: str
    video_duration: int
    summary: str
    key_points: list[str]
    keywords: list[str]

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_video(request: SummaryRequest, user_id: str = Depends(verify_jwt)):
    try:

        video_id = extract_video_id(request.youtube_url)

        transcript = get_transcript(video_id)
        if not transcript:
            raise HTTPException(400, "No transcript available")

        metadata = get_video_metadata(request.youtube_url)

        summary = generate_summary(transcript, metadata["title"], request.summary_length)
        
        return SummaryResponse(
            video_title=metadata["title"],
            video_duration=metadata["duration"],
            summary=summary["summary"],
            key_points=summary["key_points"],
            keywords=summary["keywords"]
        )
        
    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {str(e)}")

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL"""
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['id']

def get_transcript(video_id: str) -> str:
    """Get YouTube transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript_list])
    except:
        return ""

def get_video_metadata(url: str) -> dict:
    """Get video title, duration"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get('title', 'Unknown'),
            "duration": info.get('duration', 0)
        }

def generate_summary(transcript: str, title: str, length: str = "short") -> dict:
    """Generate AI summary using OpenAI/Groq"""
    
    prompt = f"""
    Summarize this YouTube video transcript: "{title}"
    
    Transcript: {transcript[:8000]}...  # Truncate for token limits
    
    Provide:
    1. SUMMARY ({length.upper()} version)
    2. KEY POINTS (bullet list, 5-8 items)
    3. KEYWORDS (5-10 most important)
    
    JSON format only:
    {{
      "summary": "...",
      "key_points": ["point1", "point2"],
      "keywords": ["kw1", "kw2"]
    }}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    
    import json
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
