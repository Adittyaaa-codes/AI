from fastapi import Security
from fastapi.security import APIKeyHeader
import re

from langchain_qdrant import QdrantVectorStore
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_user_id(api_key: str = Security(API_KEY_HEADER)) -> str:
    return api_key

def collection_name_for(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
    return f"user_{safe}_docs"

def get_vector_store_for(user_id: str) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        collection_name=collection_name_for(user_id),
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )