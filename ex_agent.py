from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os


load_dotenv()

# Create LLM with Google Search enabled
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Tool definition
@tool
def search_web_material(query: str) -> str:
    """Search the web for the best resources available on a topic 
    and extract the informations"""
    response = llm.invoke(query)
    return response.content

# Simple, clear system prompt (don't use ReAct format manually)
web_search_template = """You are an expert ExplanationAgent that helps explain complex topics.

Your approach:
1. Use web search to find the most relevant and accurate information
2. Compare multiple sources when available
3. Synthesize the information into clear, simple explanations
4. Provide examples and analogies to make concepts easier to understand

Always cite sources when using web search results."""

# Create agent
ExplanationAgent = create_agent(
    model=llm,
    tools=[search_web_material],
    system_prompt=web_search_template
)

# Invoke agent - use 'messages' key, not 'input'
response = ExplanationAgent.invoke({
    "messages": [("user", "Explain some advanced python programming concepts")]
})

# Print response
print(response['messages'][-1].content)
