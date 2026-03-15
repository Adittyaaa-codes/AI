from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import asyncio
import os

from Utils.utility import embedding_model, get_vector_store_for

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

@tool
def analyze_docs(query: str) -> str:
    """Analyze the user query, do similarity search and find relevant chunks from uploaded documents."""
    try:
        vs = get_vector_store_for()
        docs = vs.similarity_search(query=query, k=4)
        if not docs:
            return "No relevant study materials found in your uploaded documents."
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
        return context
    except Exception as e:
        return f"Could not search documents: {str(e)}. Try using web search instead."

@tool
def search_web_material(query: str) -> str:
    """Search the web for the best resources available on a topic and extract the information."""
    try:
        web_search_tool = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
        )
        return web_search_tool.invoke(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"


web_search_template = """You are an expert ExplanationAgent that helps explain complex topics \
in the easiest way possible so that a user without any prerequisite knowledge can understand easily.

Your approach:
1. ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information.
2. Use the context from analyze_docs to answer the user's question accurately.
3. If the documents don't contain sufficient information, use search_web_material for additional context.
4. Synthesize the information into clear, simple explanations with examples.
5. Always cite which source you are using (documents or web search).
6. If neither source contains the answer, respond with "Your query is out of your source materials".

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) over web search."""

ExplanationAgent = create_agent(
    model=llm,
    tools=[analyze_docs, search_web_material],
    system_prompt=web_search_template
)

if __name__ == "__main__":
    user_query = input("Enter Your Query : ")
    response = ExplanationAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response['messages'][-1].content)
