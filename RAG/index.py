from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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

# Clean each document's content
for document in doc:
    document.page_content = clean_text(document.page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(doc)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    collection_name="discrete-mathematics",
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
)

print("Indexing completed.")