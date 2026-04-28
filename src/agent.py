from typing import TypedDict, List, Optional
import os
import json

from IPython.display import Image, display


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.tool import search_tool
from src.schema import RequestInput

load_dotenv()

memory = MemorySaver()

tools = [search_tool]
tool_node = ToolNode(tools)

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

llm_with_tools = llm.bind_tools(tools)


class ResearchAgentState(TypedDict):
    user_query: RequestInput
    messages: List[BaseMessage]
    should_use_tool: bool
    generated_answer: Optional[str]


def decide_tool(state: ResearchAgentState):
    query = state["user_query"].user_query

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Decide whether a tool is needed to answer the user's question."
        ),
        HumanMessage(
            content=f"User query: {query}"
        ),
    ]

    ai_message = llm_with_tools.invoke(messages)

    should_use_tool = bool(getattr(ai_message, "tool_calls", None))

    return {
        "messages": messages + [ai_message],
        "should_use_tool": should_use_tool,
    }


def route_tool(state: ResearchAgentState):
    if state["should_use_tool"]:
        return "run_tool"
    return "give_response"


def give_answer(state: ResearchAgentState):
    request = state["user_query"]
    previous_messages = state["messages"]

    tool_outputs = []

    for msg in previous_messages:
        if isinstance(msg, ToolMessage):
            tool_name = msg.name or "unknown tool"
            tool_content = msg.content

            if isinstance(tool_content, list):
                tool_content = json.dumps(tool_content, ensure_ascii=False, default=str)

            tool_outputs.append(
                f"Tool Name: {tool_name}\nTool Output:\n{tool_content}"
            )

    compiled_tool_context = (
        "\n\n------------------------------\n\n".join(tool_outputs)
        if tool_outputs
        else "No tool outputs available."
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Answer the user's question clearly and accurately."
        ),
        HumanMessage(
            content=f"""
User query:
{request.user_query}

Tool context:
{compiled_tool_context}

Generate a useful answer.
""".strip()
        ),
    ]

    final_message = llm.invoke(messages)

    return {
        "messages": previous_messages + messages + [final_message],
        "generated_answer": final_message.content,
    }


research_agent = StateGraph(ResearchAgentState)

research_agent.add_node("decide_tools", decide_tool)
research_agent.add_node("run_tool", tool_node)
research_agent.add_node("give_response", give_answer)

research_agent.add_edge(START, "decide_tools")

research_agent.add_conditional_edges(
    "decide_tools",
    route_tool,
    {
        "run_tool": "run_tool",
        "give_response": "give_response",
    },
)

research_agent.add_edge("run_tool", "give_response")
research_agent.add_edge("give_response", END)

agent = research_agent.compile(checkpointer=memory)

png_data = agent.get_graph().draw_mermaid_png()

with open("research_agent_graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved as research_agent_graph.png")

input_state = {
    "user_query": RequestInput(user_query="CEO od spacex"),
    "messages": [],
    "should_use_tool": False,
    "generated_answer": None,
}

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(input_state, config)

print(f"Result: {result['generated_answer']}")