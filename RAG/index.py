from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import re
import os
import sys
from dotenv import load_dotenv

# Ensure we can import from Utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.utility import embedding_model, _make_qdrant_client

load_dotenv()

def clean_text(text: str) -> str:
    """Clean PDF text by removing extra whitespace and newlines"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

file_path = "DISCRETE-MATHEMATICS.pdf"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    sys.exit(1)

loader = PyPDFLoader(file_path)
doc = loader.load()

print(f"Loaded {len(doc)} pages from PDF")

# Clean each document's content
for document in doc:
    document.page_content = clean_text(document.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(doc)

print(f"Split into {len(chunks)} chunks")

from qdrant_client.models import Distance, VectorParams

client = _make_qdrant_client()
coll = "discrete-mathematics"

# models/text-embedding-004 produces 768-dim vectors.
existing = [c.name for c in client.get_collections().collections]
if coll not in existing:
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"Created collection '{coll}' with 768-dim vectors")
else:
    info = client.get_collection(coll)
    existing_size = info.config.params.vectors.size
    if existing_size != 768:
        print(f"⚠️ Dimension mismatch in '{coll}': expected 768, found {existing_size}. Recreating...")
        client.delete_collection(coll)
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Recreated collection '{coll}' with 768-dim vectors")

print("Starting indexing with Google Gemini text-embedding-004 (768 dimensions)...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    collection_name=coll,
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("✅ Indexing completed successfully.")