from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
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
        return f"Could not search documents: {str(e)}. Try generating questions from general knowledge."

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
1. ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information.
2. Use the context from analyze_docs to generate the questions and answers for the user.
3. If the documents don't contain sufficient information, generate the questions from general knowledge.
4. Synthesize the information into clear, simple answers of questions.
5. Always cite which source you are using (documents or general knowledge).

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) first."""

QAAgent = create_agent(
    model=llm,
    tools=[analyze_docs, ques_generator],
    system_prompt=qa_generator_template
)

if __name__ == "__main__":
    user_query = input("On which topic you want questions: ")
    response = QAAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response['messages'][-1].content)
