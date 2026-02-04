import os
import sys
import re
import tempfile
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import models
import os, sys, re, tempfile, uuid, uvicorn

from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import os, sys, re, tempfile, uuid
from fastapi import Header,Depends, Security
from fastapi.security import APIKeyHeader
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Langgraph.multi_agent import rag_app
from Utils.utility import get_user_id, collection_name_for,embedding_model

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
    query: str
    
class UploadResponse(BaseModel):
    message: str
    collection: str
    processed: int
    failed: int
    total_chunks: int
    details: dict
    

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
    else:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)

@app.post("/upload_docs", response_model=UploadResponse)
async def upload_docs(files: List[UploadFile] = File(...), user_id: str = Depends(get_user_id)):
    processed, failed, details = 0, 0, {}
    tmp_paths = []
    for f in files:
        try:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in [".pdf", ".txt", ".md"]:
                failed += 1
                details[f.filename] = "unsupported"
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                data = await f.read()
                tmp.write(data)
                tmp_paths.append((f.filename, tmp.name))
        except Exception as e:
            failed += 1
            details[f.filename] = str(e)
    all_docs = []
    for orig, path in tmp_paths:
        try:
            docs = _load_file_to_docs(path)
            for d in docs:
                meta = d.metadata or {}
                meta["user_id"] = user_id
                meta["source_name"] = orig
                meta["doc_id"] = str(uuid.uuid4())
                d.metadata = meta
            all_docs.extend(docs)
            processed += 1
        finally:
            try:
                os.unlink(path)
            except:
                pass
    if all_docs:
        QdrantVectorStore.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name_for(user_id),
        )
    return UploadResponse(
        message="ok",
        collection=collection_name_for(user_id),
        processed=processed,
        failed=failed,
        total_chunks=len(all_docs),
        details=details,
    )

    
@app.get("/list_docs")
async def list_documents(user_id: str = Depends(get_user_id)):
    """List all unique source documents in the collection"""
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"))
        
        records = client.scroll(
            collection_name=collection_name_for(user_id),
            limit=1000,
            with_payload=True
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
    x_user_id: str = Header(..., alias="X-User-ID")
):
    """Delete a single document by filename for a specific user"""
    
    try:
        from qdrant_client import QdrantClient, models
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"))
        collection_name = get_user_collection_name(x_user_id)
        
        try:
            client.get_collection(collection_name)
        except:
            raise HTTPException(
                status_code=404, 
                detail=f"Collection not found for user"
            )
        
        result = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filename)
                        )
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


@app.post("/chat")
async def stream_response(request: ChatRequest, user_id: str = Depends(get_user_id)):
    async def generate():
        try:
            async for event in rag_app.astream_events(
                {"messages": [HumanMessage(content=request.query)]},
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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
