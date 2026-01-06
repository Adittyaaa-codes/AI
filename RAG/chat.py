from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="discrete-mathematics",
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
)

user_query = "Explain duality law and tell the page number in the book where it is explained."

retrieved_docs = vector_store.similarity_search(
    query=user_query,
    k=3
)

context = [result.page_content for result in retrieved_docs]

SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer user queries accurately.
If the context does not contain the answer, respond with "I don't know".

Context:
{context}
"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT.format(context='\n\n'.join(context))},
        {"role": "user", "content": user_query}
    ]
)

print(response.choices[0].message.content)  