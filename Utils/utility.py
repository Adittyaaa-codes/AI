import re
import os
import contextvars
import google.generativeai as genai
import jwt
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
auth_scheme = HTTPBearer(auto_error=True)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

active_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("active_user_id", default=None)

def embed_text(text: str, task_type: str) -> list[float]:
    if not text or not text.strip():
        return [0.0] * 3072
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    embedding = result["embedding"]
    assert len(embedding) == 3072, f"Dimension mismatch: expected 3072, got {len(embedding)}"
    return embedding

def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Batch version of embed_text to avoid sequential API calls."""
    if not texts:
        return []
    
    # Handle empty strings within the batch by replacing them with a zero vector
    # This avoids API errors for empty content while still processing the batch
    processed_texts = []
    indices_to_fill_zero = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            indices_to_fill_zero.append(i)
            processed_texts.append(" ") # Replace with a non-empty string for the API call
        else:
            processed_texts.append(t)

    # AFTER
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=processed_texts,
        task_type=task_type
    )
    embeddings = result["embeddings"]

    for i in indices_to_fill_zero:
        embeddings[i] = [0.0] * 3072

    for e in embeddings:
        assert len(e) == 3072, f"Dimension mismatch: expected 3072, got {len(e)}"
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

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()
    