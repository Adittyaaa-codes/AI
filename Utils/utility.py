import re
import os
import contextvars
from fastapi import Security
from fastapi.security import APIKeyHeader
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

active_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("active_user_id", default=None)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_user_id(api_key: str = Security(API_KEY_HEADER)) -> str:
    return api_key

def collection_name_for(user_id: str | None) -> str:
    if not user_id:
        return os.getenv("QDRANT_COLLECTION", "test-collection")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
    return f"user_{safe}_docs"

def _make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False,
        timeout=20,
    )

def get_vector_store_for(user_id: str | None = None) -> QdrantVectorStore:
    effective_user_id = user_id or active_user_id.get()
    client = _make_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name_for(effective_user_id),
        embedding=embedding_model,
    )