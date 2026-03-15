import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.0-flash")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

print("Connecting to Qdrant at:", os.getenv("QDRANT_URL"))
print("Collection: discrete-mathematics")

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="discrete-mathematics",
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

user_query = "Explain duality law and tell the page number in the book where it is explained."

print(f"\n🔍 Querying Qdrant for: {user_query}")
retrieved_docs = vector_store.similarity_search(
    query=user_query,
    k=3
)

# DEBUG: Log retrieval results
print(f"\n✅ Retrieved {len(retrieved_docs)} documents")
if retrieved_docs:
    print(f"   Score of first result: {retrieved_docs[0].metadata.get('_relevance_score', 'N/A')}")
    print(f"   First result preview: {retrieved_docs[0].page_content[:100]}...")
else:
    print("   ⚠️  WARNING: No documents retrieved! Check:")
    print("     - Qdrant connection and collection name")
    print("     - Embedding model dimensions match")
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