from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

from Utils.utility import embedding_model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

_default_collection = os.getenv("QDRANT_COLLECTION", "test-collection")
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=_default_collection,
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
def ques_generator(query:str)->str:
    """Use the context from analyze_docs tool,analyze and create questions for the user 
    and make the user exam-ready"""
    
    response = llm.invoke(query)
    return response.content

def _unwrap_tool(tool_obj):
    for attr in ("func", "run", "invoke", "__wrapped__"):
        if hasattr(tool_obj, attr):
            return getattr(tool_obj, attr)
    return tool_obj

ques_generator_callable = _unwrap_tool(ques_generator)

qa_generator_template = """You are an expert QAGeneratorAgent that helps generate only relevant questions by 
analyzing the context from uploaded documents.If the query is about some explaination then you can generate questions based on the context and also provide answers to those questions.

Example:
User Query: "About Duality Law"
Your approach: The user is asking about Duality Law, so you will first use the analyze_docs tool to search for relevant information about Duality Law in the uploaded documents and only generate questions based on the context from uploaded documents, do not explain anything about duality law. If the documents don't contain sufficient information about Duality Law, then you can generate questions from web search but you must prioritize questions from uploaded documents first.
Answer: "Here are some questions based on the context from your uploaded documents about Duality Law:
1. What is the Duality Law in Boolean algebra?

You must prioritize PYQs from the source materials which you get from analyze_docs tool.

You must generate the questions in most simplest way possible such that a user without any prerequistic knowlegde can 
understand easily.

You must analyze PYQs and Current semester sources for generating questions.

Your approach:
1. ALWAYS start by using the analyze_docs tool to search the uploaded documents for relevant information
2. Use the context from analyze_docs to generate the questions and answers for user
3. If the documents don't contain sufficient information, then generate the questions from web
4. Synthesize the information into clear, simple answers of questions
5. Always cite which source you're using (documents or web search)

IMPORTANT: Prioritize information from analyze_docs (uploaded documents) first only."""

QAAgent = create_agent(
    model=llm,
    tools=[analyze_docs,ques_generator],
    system_prompt=qa_generator_template
)

if __name__ == "__main__":
    user_query = input("On which topic you want questions: ")
    response = QAAgent.invoke({
        "messages": [("user", user_query)]
    })
    print(response['messages'][-1].content)
