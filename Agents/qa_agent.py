from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os

from Utils.utility import get_vector_store_for

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

@tool
def analyze_docs(query: str, config: RunnableConfig) -> str:
    """Analyze the user query, do similarity search and find relevant chunks from uploaded documents."""
    try:
        user_id = config.get("configurable", {}).get("user_id")
        print(f"\n[QA-DEBUG] analyze_docs called | user_id={user_id} | query={query[:80]}")

        vs = get_vector_store_for(user_id)

        # Filter by user_id to ensure isolation between users
        search_filter = None
        if user_id:
            search_filter = Filter(
                must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
            )

        docs = vs.similarity_search(query=query, k=4, filter=search_filter)

        print(f"[QA-DEBUG] Retrieved {len(docs)} chunks from collection for user_id={user_id}")
        if docs:
            print(f"[QA-DEBUG] Top chunk score (metadata): {docs[0].metadata.get('_relevance_score', 'N/A')}")
            print(f"[QA-DEBUG] Top chunk payload keys: {list(docs[0].metadata.keys())}")
            print(f"[QA-DEBUG] Top chunk preview: {docs[0].page_content[:120].replace(chr(10), ' ')}...")
        else:
            print(f"[QA-DEBUG] ⚠️ 0 results — collection may be empty, wrong name, or filter is too strict.")

        if not docs:
            return "No relevant study materials found in your uploaded documents."

        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
        print(f"[QA-DEBUG] Context length being sent to LLM: {len(context)} chars")
        return context

    except Exception as e:
        print(f"[QA-DEBUG] ❌ Error in analyze_docs: {str(e)}")
        return f"Could not search documents: {str(e)}. Try generating questions from general knowledge."

@tool
def get_available_sources(query: str, config: RunnableConfig) -> str:
    """List all original document names/source materials currently stored in the user's library."""
    try:
        from Utils.utility import _make_qdrant_client, collection_name_for

        user_id = config.get("configurable", {}).get("user_id")
        client = _make_qdrant_client()
        coll = collection_name_for(user_id)
        print(f"[QA-DEBUG] get_available_sources | collection={coll} | user_id={user_id}")

        records = client.scroll(
            collection_name=coll,
            limit=100,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
            ) if user_id else None
        )

        sources = set()
        for record in records[0]:
            if record.payload and 'metadata' in record.payload:
                source = record.payload['metadata'].get('source')
                if source:
                    sources.add(source)

        print(f"[QA-DEBUG] Found {len(sources)} unique sources in collection")

        if not sources:
            return "No documents have been uploaded to your library yet."

        return "The following source materials are available in your library:\n- " + "\n- ".join(list(sources))

    except Exception as e:
        print(f"[QA-DEBUG] ❌ Error in get_available_sources: {str(e)}")
        return f"Error retrieving source list: {str(e)}"

@tool
def ques_generator(query: str) -> str:
    """Use the context from analyze_docs tool, analyze and create questions for the user and make the user exam-ready."""
    response = llm.invoke(query)
    return response.content

def _unwrap_tool(tool_obj):
    for attr in ("func", "run", "invoke", "__wrapped__"):
        if hasattr(tool_obj, attr):
            return getattr(tool_obj, attr)
    return tool_obj

ques_generator_callable = _unwrap_tool(ques_generator)

qa_generator_template = """You are an expert QAGeneratorAgent that helps generate only relevant questions by \
analyzing the context from uploaded documents. If the query is about some explanation then you can generate questions \
based on the context and also provide answers to those questions.

Your approach:
1. If the user asks what documents/sources are available, ALWAYS use the get_available_sources tool.
2. For all other queries, ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information.
3. Use the context from analyze_docs to generate the questions and answers for the user.
4. If the documents don't contain sufficient information, generate the questions from general knowledge.
5. Synthesize the information into clear, simple answers of questions.
6. Always cite which source you are using (documents or general knowledge).

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) first."""

QAAgent = create_agent(
    model=llm,
    tools=[analyze_docs, get_available_sources, ques_generator],
    system_prompt=qa_generator_template
)

if __name__ == "__main__":
    user_query = input("On which topic you want questions: ")
    response = QAAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response['messages'][-1].content)
