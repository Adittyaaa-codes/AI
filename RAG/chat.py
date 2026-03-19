import os
import sys
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

# Ensure we can import from Utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.utility import embedding_model, _make_qdrant_client

load_dotenv()

from google import generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.0-flash")

print("Connecting to Qdrant at:", os.getenv("QDRANT_URL"))
print("Collection: discrete-mathematics")

client = _make_qdrant_client()
vector_store = QdrantVectorStore(
    client=client,
    collection_name="discrete-mathematics",
    embedding=embedding_model,
)

user_query = "Explain duality law and tell the page number in the book where it is explained."

print(f"\n🔍 Querying Qdrant for: {user_query}")
# Using the embedding_model wrapper which internally uses task_type="retrieval_query" for searches
retrieved_docs = vector_store.similarity_search(
    query=user_query,
    k=3
)

# DEBUG: Log retrieval results
print(f"\n✅ Retrieved {len(retrieved_docs)} documents")
if retrieved_docs:
    # Note: similarity_search might not return score in metadata by default unless using similarity_search_with_score
    print(f"   First result preview: {retrieved_docs[0].page_content[:100]}...")
else:
    print("   ⚠️  WARNING: No documents retrieved! Check:")
    print("     - Qdrant connection and collection name")
    print("     - Embedding model dimensions match (should be 768)")
    print("     - Collection is not empty")

context = [result.page_content for result in retrieved_docs]

SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer user queries accurately.
If the context does not contain the answer, respond with "I don't know".

Context:
{context}
"""

response = llm_model.generate_content(
    SYSTEM_PROMPT.format(context='\n\n'.join(context)) + "\n\nUser Query: " + user_query
)

print("\n" + response.text)