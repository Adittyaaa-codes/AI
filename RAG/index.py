from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import os
from dotenv import load_dotenv

load_dotenv()

file_path = "DISCRETE-MATHEMATICS.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()

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