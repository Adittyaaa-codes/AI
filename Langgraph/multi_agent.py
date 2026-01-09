from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.ex_agent import ExplanationAgent
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
    """Route to appropriate agent based on user intent"""
    intent = state.get("user_intent", "").lower()
    
    if "question" in intent or "quiz" in intent or "test" in intent or "practice" in intent:
        return "generate_questions"
    else:
        return "explain"

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

app = workflow.compile()

if __name__ == "__main__":
    print("\n=== Multi-Agent Learning Assistant ===")
    print("I can help you:")
    print("1. Understand concepts (explanation)")
    print("2. Generate practice questions\n")
    
    user_intent = input("What would you like? (understand/questions): ")
    user_query = input("Enter your topic: ")
    
    result = app.invoke({
        "messages": [HumanMessage(content=user_query)],
        "user_intent": user_intent,
        "query": user_query
    })
    
    print("\n=== Response ===")
    print(result["messages"][-1].content)
