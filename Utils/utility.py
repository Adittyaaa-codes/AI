import re
import os
import contextvars
from fastapi import Security
from fastapi.security import APIKeyHeader
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Global context for the current user ID
active_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("active_user_id", default=None)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_user_id(api_key: str = Security(API_KEY_HEADER)) -> str:
    return api_key

def collection_name_for(user_id: str | None) -> str:
    if not user_id:
        return os.getenv("QDRANT_COLLECTION", "test-collection")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
    return f"user_{safe}_docs"

def get_vector_store_for(user_id: str | None = None) -> QdrantVectorStore:
    # Use provided user_id, or fall back to the context variable
    effective_user_id = user_id or active_user_id.get()
    
    return QdrantVectorStore.from_existing_collection(
        collection_name=collection_name_for(effective_user_id),
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )