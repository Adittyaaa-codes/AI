from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Langgraph.multi_agent import rag_app

app = FastAPI()

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

@app.get("/")
async def root():
    return {"message": "RAG Multi-Agent API is running"}
 