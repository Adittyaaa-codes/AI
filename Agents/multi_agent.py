from langgraph.graph import StateGraph, END,START
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.ex_agent import ExplanationAgent, llm
from Agents.qa_agent import QAAgent

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    user_intent: str
    query: str
    user_id: str

def explanation_node(state: AgentState):
    result = ExplanationAgent.invoke(state)
    return {"messages": result["messages"]}

def qa_generation_node(state: AgentState):
    try:
        user_id = state.get("user_id")
    except Exception:
        user_id = None

    if user_id:
        try:
            from Utils.utility import get_vector_store_for
            vs = get_vector_store_for(user_id)
            query = state.get("query") or ""
            if not query and state.get("messages"):
                last = state.get("messages")[-1]
                if isinstance(last, BaseMessage):
                    query = getattr(last, "content", "")

            docs = []
            if query:
                docs = vs.similarity_search(query=query, k=5)

            context = "\n\n".join([d.page_content for d in docs]) if docs else ""

            messages = list(state.get("messages", []))
            if context:
                messages.insert(0, HumanMessage(content=f"CONTEXT:\n{context}"))

            new_state = dict(state)
            new_state["messages"] = messages

            from Agents.qa_agent import ques_generator_callable as ques_generator
            qcount = 5
            prompt = f"Generate {qcount} practice questions with answers about: {query}.\nContext:\n{context}"
            questions = ques_generator(prompt)
            return {"messages": [HumanMessage(content=questions)]}
        except Exception:
            from Agents.qa_agent import ques_generator_callable as ques_generator
            qcount = 5
            prompt = f"Generate {qcount} practice questions with answers about: {state.get('query') or ''}."
            questions = ques_generator(prompt)
            return {"messages": [HumanMessage(content=questions)]}

    from Agents.qa_agent import ques_generator_callable as ques_generator
    qcount = 5
    q = state.get("query") or ""
    prompt = f"Generate {qcount} practice questions with answers about: {q}."
    questions = ques_generator(prompt)
    return {"messages": [HumanMessage(content=questions)]}


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
        print("Ask me anything!!Here agents can:-")
        print("- Explain anything from your resources ")
        print("- Generate questions analyzing your sources(PYQs)\n")
        
        user_query = input("Enter your query: ")
        
        print("\n=== Response ===\n")

        async for event in rag_app_ex.astream_events(
            {"messages": [HumanMessage(content=user_query)]},
            version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
        
        print("\n\n=== Done ===")
    
    asyncio.run(main())


