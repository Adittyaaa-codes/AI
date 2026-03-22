from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os

from Utils.utility import get_vector_store_for

load_dotenv()

# Fixed stable model for Gemini AI Studio.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True,
    temperature=0,
)

@tool
def analyze_docs(query: str, config: RunnableConfig) -> str:
    """Analyze the user query, do similarity search and find relevant chunks from uploaded documents."""
    try:
        user_id = config.get("configurable", {}).get("user_id")
        vs = get_vector_store_for(user_id)

        search_filter = None
        if user_id:
            search_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )

        docs = vs.similarity_search(query=query, k=4, filter=search_filter)

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
def get_available_sources(query: str, config: RunnableConfig) -> str:
    """List all original document names/source materials currently stored in the user's library."""
    try:
        from Utils.utility import _make_qdrant_client, collection_name_for

        user_id = config.get("configurable", {}).get("user_id")
        client = _make_qdrant_client()
        coll = collection_name_for(user_id)

        records = client.scroll(
            collection_name=coll,
            limit=100,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ) if user_id else None
        )

        sources = set()
        for record in records[0]:
            if record.payload:
                source = record.payload.get('source')
                if source:
                    sources.add(source)

        if not sources:
            return "No documents have been uploaded to your library yet."

        return "The following source materials are available in your library:\n- " + "\n- ".join(list(sources))

    except Exception as e:
        return f"Error retrieving source list: {str(e)}"

@tool
def search_web_material(query: str) -> str:
    """Search the web for the best resources available on a topic and extract the information."""
    try:
        web_search_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
        return web_search_tool.invoke(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"

system_prompt = """You are an expert ExplanationAgent that helps explain complex topics \
in the easiest way possible so that a user without any prerequisite knowledge can understand easily.

Your approach:
1. If the user asks what documents/sources are available, ALWAYS use the get_available_sources tool.
2. For all other queries, ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information.
3. Use the context from analyze_docs to answer the user's question accurately.
4. If the documents don't contain sufficient information, use search_web_material for additional context.
5. Synthesize the information into clear, simple explanations with examples.
6. Always cite which source you are using (documents or web search).
7. If neither source contains the answer, respond with "Your query is out of your source materials".

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) over web search."""

tools = [analyze_docs, get_available_sources, search_web_material]

ExplanationAgent = create_react_agent(llm, tools, prompt=system_prompt)

if __name__ == "__main__":
    user_query = input("Enter Your Query : ")
    response = ExplanationAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response["messages"][-1].content)
