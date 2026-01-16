from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.ex_agent import ExplanationAgent, llm
from Agents.qa_agent import QAAgent

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_intent: str
    query: str

def explanation_node(state: AgentState):
    """Explanation agent for understanding concepts"""
    result = ExplanationAgent.invoke(state)
    return {"messages": result["messages"]}

def qa_generation_node(state: AgentState):
    """QA generation agent for creating practice questions"""
    result = QAAgent.invoke(state)
    return {"messages": result["messages"]}

def route_based_on_intent(state: AgentState) -> Literal["explain", "generate_questions"]:
    """Route to appropriate agent based on LLM analysis of user intent"""
    user_query = state["messages"][0].content

    intent_prompt = f"""Analyze the following user query and determine the intent.
    
    User Query: "{user_query}"

    Respond with ONLY ONE WORD:
    - "explaination : " if the user wants to understand, learn, or get an explanation of a topic
    - "questions : " if the user wants practice questions, quiz, test, or exam preparation

    Intent:"""
    
    response = llm.invoke(intent_prompt)
    intent = response.content.strip().lower()
    
    if "question" in intent or intent == "questions":
        return "generate_questions"
    elif "explain" in intent or intent == "explain":
        return "explain"
    else:
        return None

workflow = StateGraph(AgentState)

workflow.add_node("explanation_agent", explanation_node)
workflow.add_node("qa_agent", qa_generation_node)

workflow.add_conditional_edges(
    "__start__",  
    route_based_on_intent,
    {
        "explain": "explanation_agent",
        "generate_questions": "qa_agent"
    }
)

workflow.add_edge("explanation_agent", END)
workflow.add_edge("qa_agent", END)

rag_app = workflow.compile()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("\n=== Multi-Agent Learning Assistant ===")
        print("Ask me anything!!Here agents can:-")
        print("- Explain anything from your resources ")
        print("- Generate questions analyzing your sources(PYQs)\n")
        
        user_query = input("Enter your query: ")
        
        print("\n=== Response ===\n")
        
        # Token-by-token streaming
        async for event in app.astream_events(
            {"messages": [HumanMessage(content=user_query)]},
            version="v2"
        ):
            kind = event["event"]
            
            # Stream tokens from LLM
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
        
        print("\n\n=== Done ===")
    
    asyncio.run(main())


