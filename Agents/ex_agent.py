from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="discrete-mathematics",
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
)
@tool
def analyze_docs(query: str)->str:
    """Analyze the user query and do similarity search and find relevant chunks"""
    
    docs = vector_store.similarity_search(query=query)
    if not docs:
        return "No relevant study materials found in your uploaded documents."
    
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    ])
    
    return context


@tool
def search_web_material(query: str) -> str:
    """Search the web for the best resources available on a topic 
    and extract the informations"""
    response = llm.invoke(query)
    return response.content

web_search_template = """You are an expert ExplanationAgent that helps explain complex topics 
in most easiest way possible such that a user without any prerequistic knowlegde can 
understand easily.

Your approach:
1. ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information
2. Use the context from analyze_docs to answer the user's question accurately
3. If the documents don't contain sufficient information, then use search_web_material for additional context
4. Synthesize the information into clear, simple explanations with examples
5. Always cite which source you're using (documents or web search)

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) over web search."""

ExplanationAgent = create_agent(
    model=llm,
    tools=[analyze_docs,search_web_material],
    system_prompt=web_search_template
)

if __name__ == "__main__":
    user_query = input("Enter Your Query : ")
    response = ExplanationAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response['messages'][-1].content)


