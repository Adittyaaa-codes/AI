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

from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Langgraph.multi_agent import rag_app

def clean_text(text: str) -> str:
    """Clean PDF text by removing extra whitespace and newlines"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
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
    query: str
    
class UploadResponse(BaseModel):
    message: str
    collection: str
    processed: int
    failed: int
    total_chunks: int
    details: dict
    
def get_document_loader(file_path: str, file_extension: str):
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    else:
        return TextLoader(file_path, encoding='utf-8')

async def process_uploaded_file(file: UploadFile) -> dict:
    allowed_extensions = ['.pdf', '.txt', '.md']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        return {
            "filename": file.filename,
            "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
            "status": "failed"
        }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        loader = get_document_loader(temp_file_path, file_extension)
        documents = loader.load()
        
        for document in documents:
            document.page_content = clean_text(document.page_content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        
        chunks = text_splitter.split_documents(documents)
        
        vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        collection_name="test-collection",
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        )
        
        for chunk in chunks:
            chunk.metadata['source'] = file.filename
        
        vector_store.add_documents(documents=chunks)
        
        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename": file.filename,
            "error": str(e),
            "status": "failed"
        }
    
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/")
async def root():
    return {
        "message": "RAG Multi-Agent API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": "connected",
        "agents": "ready"
    }


@app.post("/upload_docs", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(..., description="PDF, TXT, or MD files")):
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    for file in files:
        result = await process_uploaded_file(file)
        results.append(result)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    total_chunks = sum(r.get("chunks", 0) for r in successful)
    
    return {
        "message": "Document upload completed",
        "collection": "test-collection",
        "processed": len(successful),
        "failed": len(failed),
        "total_chunks": total_chunks,
        "details": {
            "successful": successful,
            "failed": failed
        }
    }
    
@app.get("/list_docs")
async def list_documents():
    """List all unique source documents in the collection"""
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=os.getenv("QDRANT_URL"))
        
        records = client.scroll(
            collection_name="test-collection",
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


@app.delete("/delete_docs")
async def delete_multiple_documents(filenames: List[str]):
    """Delete multiple documents by their filenames"""
    
    deleted = []
    failed = []
    
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    
    for filename in filenames:
        try:
            result = client.delete(
                collection_name="test-collection",
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
            deleted.append(filename)
        except Exception as e:
            failed.append({"filename": filename, "error": str(e)})
    
    return {
        "deleted": deleted,
        "failed": failed,
        "total_deleted": len(deleted)
    }

@app.post("/chat")
async def stream_response(request: ChatRequest):
    
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
