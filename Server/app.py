from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Langgraph.multi_agent import rag_app

from RAG.index import vector_store,embedding_model,clean_text


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Multi-Agent API is running"}



@app.post("/upload_docs")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Endpoint to upload documents for RAG system"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    failed_files = []
    total_chunks = 0
    
    for file in files:
        try:
            # Validate file type
            allowed_extensions = ['.pdf', '.txt', '.md']
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                failed_files.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
                })
                continue
            
            import tempfile
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_file_path = temp_file.name
            
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            
            try:
                # Load document based on file type
                if file_extension == '.pdf':
                    loader = PyPDFLoader(temp_file_path)
                else:  # .txt or .md
                    loader = TextLoader(temp_file_path, encoding='utf-8')
                
                documents = loader.load()
                
                # Clean each document's content (same as your indexing code)
                for document in documents:
                    document.page_content = clean_text(document.page_content)
                    
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                # Split documents with your exact settings
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=400
                )
                chunks = text_splitter.split_documents(documents)
                
                # Add source metadata
                for chunk in chunks:
                    chunk.metadata['source'] = file.filename
                
                # Add documents to existing Qdrant collection
                vector_store.add_documents(documents=chunks)
                
                total_chunks += len(chunks)
                processed_files.append({
                    "filename": file.filename,
                    "chunks": len(chunks),
                    "status": "success"
                })
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })
        return {
            "message": "Document upload completed",
            "collection": "discrete-mathematics",
            "processed": len(processed_files),
            "failed": len(failed_files),
            "total_chunks": total_chunks,
            "details": {
                "successful": processed_files,
                "failed": failed_files
            }
        }

async def upload_documents():
    """Endpoint to upload documents for RAG system"""
    # Implementation for document upload goes here
    return {"message": "Document upload endpoint (to be implemented)"}

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def stream_response(request: ChatRequest):
    """Stream AI response for the user query"""
    
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
    
    return StreamingResponse(generate(), media_type="text/plain")

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
    