from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import re
import os
from dotenv import load_dotenv

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
loader = PyPDFLoader(file_path)
doc = loader.load()

print(f"Loaded {len(doc)} pages from PDF")

# Clean each document's content
for document in doc:
    document.page_content = clean_text(document.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(doc)

print(f"Split into {len(chunks)} chunks")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
coll = "discrete-mathematics"

# gemini-embedding-001 produces 3072-dim vectors.
existing = [c.name for c in client.get_collections().collections]
if coll not in existing:
    client.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"Created collection '{coll}' with 3072-dim vectors")
else:
    info = client.get_collection(coll)
    existing_size = info.config.params.vectors.size
    if existing_size != 3072:
        print(f"⚠️  Dimension mismatch in '{coll}': expected 3072, found {existing_size}. Recreating...")
        client.delete_collection(coll)
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

print("Starting indexing with Google Gemini embeddings (3072 dimensions)...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    collection_name=coll,
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("✅ Indexing completed successfully.")