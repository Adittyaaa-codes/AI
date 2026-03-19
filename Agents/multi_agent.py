from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.ex_agent import ExplanationAgent
from Agents.qa_agent import QAAgent

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    user_intent: str
    query: str
    user_id: str

def explanation_node(state: AgentState):
    user_id = state.get("user_id")
    # create_react_agent graph returns a dict with the full 'messages' list
    result = ExplanationAgent.invoke(
        {"messages": state.get("messages", [])},
        config={"configurable": {"user_id": user_id}},
    )
    # Extract only the newly generated messages (tool calls, tool responses, final answer) to avoid duplication
    original_count = len(state.get("messages", []))
    return {"messages": result["messages"][original_count:]}

def qa_generation_node(state: AgentState):
    user_id = state.get("user_id")
    result = QAAgent.invoke(
        {"messages": state.get("messages", [])},
        config={"configurable": {"user_id": user_id}},
    )
    original_count = len(state.get("messages", []))
    return {"messages": result["messages"][original_count:]}

workflow_ex = StateGraph(AgentState)
workflow_ex.add_node("explanation_agent", explanation_node)
workflow_ex.add_edge(START, "explanation_agent")
workflow_ex.add_edge("explanation_agent", END)

workflow_qa = StateGraph(AgentState)
workflow_qa.add_node("qa_agent", qa_generation_node)
workflow_qa.add_edge(START, "qa_agent")
workflow_qa.add_edge("qa_agent", END)

rag_app_qa = workflow_qa.compile()
rag_app_ex = workflow_ex.compile()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("\n=== Multi-Agent Learning Assistant ===")
        user_query = input("Enter your query: ")
        
        async for event in rag_app_ex.astream_events(
            {"messages": [HumanMessage(content=user_query)]},
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
        print("\n=== Done ===")
    
    asyncio.run(main())
