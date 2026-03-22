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
from langchain_qdrant import QdrantVectorStore

from fastapi import Header,Depends, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(find_dotenv())
print("--- STARTING AI SERVER WITH GEMINI-2.5-FLASH ---")

from Agents.multi_agent import rag_app_ex, rag_app_qa
from Utils.utility import (
    get_user_id, 
    collection_name_for, 
    embedding_model, 
    active_user_id, 
    _load_file_to_docs, 
    _unique_save_path, 
    _extract_query, 
    verify_jwt, 
    clean_text,
    auth_scheme
)

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



@app.get("/")
async def root():
    return {
        "message": "RAG Multi-Agent API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/upload_docs", response_model=UploadResponse)
async def upload_docs(
    files: List[UploadFile] = File(...), 
    subject: str = Form(...),
    chapter: str = Form(...),
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
                failed = int(failed) + 1
                details[f.filename] = "unsupported"
                continue
            save_path = _unique_save_path(base_dir, os.path.basename(f.filename))
            data = await f.read()
            with open(save_path, "wb") as out:
                out.write(data)
            tmp_paths.append((f.filename, save_path))
        except Exception as e:
            failed = int(failed) + 1
            details[f.filename] = str(e)
    all_docs = []
    for orig, path in tmp_paths:
        try:
            docs = _load_file_to_docs(path)
            if not docs:
                failed = int(failed) + 1
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
            processed = int(processed) + 1
        except Exception as e:
            failed = int(failed) + 1
            details[orig] = f"processing error: {str(e)}"
        finally:
            pass
    if all_docs:
        def index_in_background(docs, uid):
            from Utils.utility import _make_qdrant_client
            from qdrant_client.models import Distance, VectorParams
            try:
                coll = collection_name_for(uid)
                client = _make_qdrant_client()

                # gemini-embedding-001 produces 3072-dim vectors.
                # Recreate collection only if it doesn't exist with correct config.
                existing = [c.name for c in client.get_collections().collections]
                if coll not in existing:
                    client.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                    )
                    print(f"[INDEX] Created collection '{coll}' with 3072-dim cosine vectors")
                else:
                    info = client.get_collection(coll)
                    existing_size = info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else "unknown"
                    print(f"[INDEX] Collection '{coll}' already exists. Vector size: {existing_size}")
                    if existing_size != 3072:
                        print(f"[INDEX] ⚠️ DIMENSION MISMATCH: expected 3072, found {existing_size}. Recreating...")
                        client.delete_collection(coll)
                        client.create_collection(
                            collection_name=coll,
                            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                        )
                        print(f"[INDEX] Recreated collection '{coll}' with 3072-dim vectors")

                from qdrant_client.models import PointStruct
                import uuid
                from Utils.utility import embed_texts
                
                texts = [d.page_content for d in docs]
                payloads = [{"text": d.page_content, **d.metadata} for d in docs]
                embeddings = embed_texts(texts, "retrieval_document")
                
                points = [
                    PointStruct(id=str(uuid.uuid4()), vector=e, payload=p)
                    for e, p in zip(embeddings, payloads)
                ]
                
                client.upload_points(
                    collection_name=coll,
                    points=points
                )
                
                # STEP 1 — VERIFY INDEXING
                points_count = client.count(collection_name=coll).count
                print(f"[INDEX-DIAGNOSTIC] Total points in collection {coll}: {points_count}")
                if points_count > 0:
                    scroll_result = client.scroll(collection_name=coll, limit=1, with_payload=True)
                    records = scroll_result[0]
                    if records:
                        print(f"[INDEX-DIAGNOSTIC] First point payload: {records[0].payload}")
                else:
                    print(f"[INDEX-DIAGNOSTIC] ❌ CRITICAL: Upsert yielded 0 points!")
                
                print(f"✅ Successfully indexed {len(docs)} chunks for user {uid} into collection '{coll}'")
            except Exception as e:
                import traceback
                print(f"❌ Error in background indexing for user {uid}: {str(e)}")
                traceback.print_exc()

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
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        
        records = client.scroll(
            collection_name=collection_name_for(user_id),
            limit=1000,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
        )
        
        sources = set()
        for record in records[0]:
            if record.payload:
                source = record.payload.get('source')
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
                            key="source",
                            match=models.MatchValue(value=filename)
                        ),
                        models.FieldCondition(
                            key="user_id",
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



@app.post("/chat/qa")
async def stream_qa(req: Request, user_id: str = Depends(verify_jwt)):
    active_user_id.set(user_id)
    body = await req.json()
    query = (body or {}).get("query") or _extract_query(body)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    async def generate():
        try:
            from Utils.utility import _make_qdrant_client, collection_name_for, embed_text
            import google.generativeai as genai
            import os
            import json
            
            client = _make_qdrant_client()
            coll = collection_name_for(user_id)
            query_vector = embed_text(query, "retrieval_query")
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
            
            try:
                results = client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=4,
                    with_payload=True
                ).points
            except Exception as e:
                results = []
                
            print(f"Number of results returned: {len(results)}")
            if results:
                print(f"Score of top result: {results[0].score}")
                print(f"First 100 characters of top result payload text: {results[0].payload.get('text', '')[:100]}")
            
            # STEP 4 — VERIFY CONTEXT INJECTION (PART 1)
            print(f"[CHAT-DIAGNOSTIC] Chunks retrieved: {len(results)}")
            
            if not len(results):
                context_str = "No relevant context found."
            else:
                chunks = [r.payload["text"] for r in results if r.payload and "text" in r.payload]
                context_str = "\n\n".join(chunks)
                
            # STEP 4 — VERIFY CONTEXT INJECTION (PART 2)
            print(f"[CHAT-DIAGNOSTIC] Full context string: {context_str[:500]}...")
            prompt = f"Context:\n{context_str}\n\nAnswer the following Question: {query}"
            print(f"[CHAT-DIAGNOSTIC] Final prompt: {prompt[:500]}...")
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Since the frontend uses simple string concatenation (aiText += chunk), 
            # we just yield the exact text chunks to match their current logic.
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
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
async def stream_explain(request: Request, user_id: str = Depends(verify_jwt)):
    active_user_id.set(user_id)
    body = await request.json()
    query = (body or {}).get("query") or _extract_query(body)
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
        
    async def generate():
        try:
            from Utils.utility import _make_qdrant_client, collection_name_for, embed_text
            import google.generativeai as genai
            import os
            import json
            
            client = _make_qdrant_client()
            coll = collection_name_for(user_id)
            query_vector = embed_text(query, "retrieval_query")
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
            
            try:
                results = client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=4,
                    with_payload=True
                ).points
            except Exception as e:
                results = []
                
            print(f"Number of results returned: {len(results)}")
            if results:
                print(f"Score of top result: {results[0].score}")
                print(f"First 100 characters of top result payload text: {results[0].payload.get('text', '')[:100]}")
            
            # STEP 4 — VERIFY CONTEXT INJECTION (PART 1)
            print(f"[EXPLAIN-DIAGNOSTIC] Chunks retrieved: {len(results)}")
            
            if not len(results):
                context_str = "No relevant context found."
            else:
                chunks = [r.payload["text"] for r in results if r.payload and "text" in r.payload]
                context_str = "\n\n".join(chunks)
                
            # STEP 4 — VERIFY CONTEXT INJECTION (PART 2)
            print(f"[EXPLAIN-DIAGNOSTIC] Full context string: {context_str[:500]}...")
            prompt = f"Context:\n{context_str}\n\nExplain the following Question clearly: {query}"
            print(f"[EXPLAIN-DIAGNOSTIC] Final prompt: {prompt[:500]}...")
            
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Since the frontend uses simple string concatenation (aiText += chunk), 
            # we just yield the exact text chunks to match their current logic.
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
