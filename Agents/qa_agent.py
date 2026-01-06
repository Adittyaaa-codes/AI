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
def ques_generator(query:str)->str:
    """Use the context from analyze_docs tool,analyze and create questions for the user 
    and make the user exam-ready"""
    
    response = llm.invoke(query)
    return response.content

user_query = input("On which topic you want questions: ")

qa_generator_template = """You are an expert QAGeneratorAgent that helps generate relevant questions.

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

response = QAAgent.invoke({
    "messages": [("user", user_query)]
})

print(response['messages'][-1].content)
