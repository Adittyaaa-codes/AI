import re
import os
import contextvars
import google.generativeai as genai
from fastapi import Security
from fastapi.security import APIKeyHeader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

active_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("active_user_id", default=None)

def embed_text(text: str, task_type: str) -> list[float]:
    """
    Generate embeddings using Google Gemini's text-embedding-004 model.
    task_type should be "retrieval_document" for indexing and "retrieval_query" for search.
    """
    if not text or not text.strip():
        # Return zero vector if text is empty to avoid API errors
        return [0.0] * 768
        
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type=task_type
    )
    embedding = result["embedding"]
    assert len(embedding) == 768, f"Dimension mismatch: expected 768, got {len(embedding)}"
    return embedding

def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Batch version of embed_text to avoid sequential API calls."""
    if not texts:
        return []
    
    # Handle potentially large batches by splitting if necessary, 
    # though genai.embed_content usually handles reasonable lists.
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type=task_type
    )
    embeddings = result["embeddings"]
    for e in embeddings:
        assert len(e) == 768, f"Dimension mismatch: expected 768, got {len(e)}"
    return embeddings

class GeminiEmbeddings(Embeddings):
    """
    LangChain compatible wrapper for the Gemini embedding function.
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use batch embedding for efficiency
        return embed_texts(texts, "retrieval_document")

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text, "retrieval_query")

embedding_model = GeminiEmbeddings()

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
        timeout=60,  # Increased timeout for production stability
    )

def get_vector_store_for(user_id: str | None = None) -> QdrantVectorStore:
    effective_user_id = user_id or active_user_id.get()
    client = _make_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name_for(effective_user_id),
        embedding=embedding_model,
    )