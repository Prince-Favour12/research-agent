from typing import TypedDict, List, Optional, Annotated
import os
import json
import operator

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

MAX_RESEARCH_RETRIES = 3


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class MultiAgentState(TypedDict):
    # Input
    user_query: RequestInput

    # Planning Agent output
    intent: Optional[str]
    structured_plan: Optional[str]

    # Supervisor output
    refined_query: Optional[str]
    quality_feedback: Optional[str]

    # Research Agent
    # NOTE: must be named `messages` — ToolNode looks for this key specifically
    messages: List[BaseMessage]
    generated_answer: Optional[str]

    # Control
    retry_count: int
    quality_approved: bool


# ─────────────────────────────────────────────
# Planning Agent
# ─────────────────────────────────────────────

def planning_agent(state: MultiAgentState) -> dict:
    """
    Understands the user's intent and creates a structured research plan.
    """
    raw_query = state["user_query"].user_query

    messages = [
        SystemMessage(content="""You are a Planning Agent. Your job is to:
1. Understand the user's true intent behind their query (even if poorly worded).
2. Identify what kind of information is needed (facts, comparisons, explanations, etc.).
3. Break down the query into a clear, structured research plan.

Respond in this exact JSON format:
{
  "intent": "A one-sentence description of what the user really wants",
  "structured_plan": "A step-by-step breakdown of what needs to be researched to satisfy this intent"
}"""),
        HumanMessage(content=f"User query: {raw_query}"),
    ]

    response = llm.invoke(messages)

    try:
        cleaned = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent", "")
        structured_plan = parsed.get("structured_plan", "")
    except Exception:
        intent = f"Find information about: {raw_query}"
        structured_plan = f"Research the following: {raw_query}"

    print(f"\n[Planning Agent]\n  Intent: {intent}\n  Plan: {structured_plan}")

    return {
        "intent": intent,
        "structured_plan": structured_plan,
    }


# ─────────────────────────────────────────────
# Supervisor Agent — Query Refinement
# ─────────────────────────────────────────────

def supervisor_refine_query(state: MultiAgentState) -> dict:
    """
    Takes the planning output and crafts an optimised query for the Research Agent.
    Also incorporates prior quality feedback on retries.
    """
    intent = state["intent"]
    plan = state["structured_plan"]
    feedback = state.get("quality_feedback") or ""
    retry = state.get("retry_count", 0)

    feedback_section = (
        f"\n\nPrevious research was rejected. Feedback:\n{feedback}\n"
        f"This is retry #{retry}. Adjust the query to address the feedback."
        if feedback and retry > 0
        else ""
    )

    messages = [
        SystemMessage(content="""You are a Supervisor Agent. Your job is to craft the most effective search/research query for a Research Agent.
Given a research intent and plan, produce a single, clear, specific query that will yield the best results.
Consider: specificity, key entities, time relevance, and what sources would be authoritative.
Respond with ONLY the refined query — no explanation."""),
        HumanMessage(content=f"""Intent: {intent}

Research Plan:
{plan}{feedback_section}

Write the refined query:"""),
    ]

    response = llm.invoke(messages)
    refined_query = response.content.strip()

    print(f"\n[Supervisor — Refine]\n  Refined Query: {refined_query}")

    return {
        "refined_query": refined_query,
        "messages": [],   # reset for this attempt
        "generated_answer": None,
        "quality_approved": False,
    }


# ─────────────────────────────────────────────
# Research Agent — Tool Decision
# ─────────────────────────────────────────────

def research_decide_tools(state: MultiAgentState) -> dict:
    refined_query = state["refined_query"]

    messages = [
        SystemMessage(content="You are a Research Agent. Use available tools to gather accurate, up-to-date information for the query."),
        HumanMessage(content=f"Research this query thoroughly: {refined_query}"),
    ]

    ai_message = llm_with_tools.invoke(messages)
    should_use_tool = bool(getattr(ai_message, "tool_calls", None))

    print(f"\n[Research Agent]\n  Using tools: {should_use_tool}")

    return {
        "messages": messages + [ai_message],
    }


def route_research_tool(state: MultiAgentState):
    messages = state["messages"]
    last = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "run_tool"
    return "synthesise"


# ─────────────────────────────────────────────
# Research Agent — Synthesise Answer
# ─────────────────────────────────────────────

def research_synthesise(state: MultiAgentState) -> dict:
    intent = state["intent"]
    refined_query = state["refined_query"]
    previous_messages = state["messages"]

    tool_outputs = []
    for msg in previous_messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False, default=str)
            tool_outputs.append(f"Tool: {msg.name or 'unknown'}\nOutput:\n{content}")

    tool_context = (
        "\n\n---\n\n".join(tool_outputs) if tool_outputs else "No external tool data available."
    )

    messages = [
        SystemMessage(content="""You are a Research Agent. Synthesise the tool outputs into a comprehensive, accurate, well-structured answer.
Be factual. Cite sources when possible. Structure the answer clearly with key findings."""),
        HumanMessage(content=f"""User intent: {intent}
Research query: {refined_query}

Tool context:
{tool_context}

Provide a thorough answer:"""),
    ]

    final_message = llm.invoke(messages)

    print(f"\n[Research Agent — Synthesise]\n  Answer preview: {final_message.content[:120]}...")

    return {
        "messages": previous_messages + messages + [final_message],
        "generated_answer": final_message.content,
    }


# ─────────────────────────────────────────────
# Supervisor Agent — Quality Check
# ─────────────────────────────────────────────

def supervisor_quality_check(state: MultiAgentState) -> dict:
    """
    Reviews the research answer. Approves or sends it back with feedback.
    """
    intent = state["intent"]
    answer = state["generated_answer"]
    retry = state.get("retry_count", 0)

    messages = [
        SystemMessage(content="""You are a Supervisor Agent performing a quality check on a research answer.
Evaluate whether the answer:
1. Directly addresses the user's intent
2. Is accurate and well-supported
3. Is comprehensive and clearly structured
4. Has no obvious gaps or errors

Respond in this exact JSON format:
{
  "approved": true/false,
  "feedback": "If not approved, specific actionable feedback for the Research Agent. If approved, empty string."
}"""),
        HumanMessage(content=f"""User intent: {intent}

Research answer:
{answer}

Evaluate this answer:"""),
    ]

    response = llm.invoke(messages)

    try:
        cleaned = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        approved = parsed.get("approved", False)
        feedback = parsed.get("feedback", "")
    except Exception:
        approved = True
        feedback = ""

    print(f"\n[Supervisor — Quality Check]\n  Approved: {approved}\n  Feedback: {feedback or 'N/A'}")

    return {
        "quality_approved": approved,
        "quality_feedback": feedback,
        "retry_count": retry + (0 if approved else 1),
    }


def route_quality(state: MultiAgentState):
    if state["quality_approved"]:
        return "done"
    if state.get("retry_count", 0) >= MAX_RESEARCH_RETRIES:
        print(f"\n[Supervisor] Max retries ({MAX_RESEARCH_RETRIES}) reached. Accepting best answer.")
        return "done"
    return "retry"


# ─────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────

graph = StateGraph(MultiAgentState)

# Nodes
graph.add_node("planning_agent", planning_agent)
graph.add_node("supervisor_refine", supervisor_refine_query)
graph.add_node("research_decide_tools", research_decide_tools)
graph.add_node("run_tool", tool_node)
graph.add_node("research_synthesise", research_synthesise)
graph.add_node("supervisor_quality", supervisor_quality_check)

# Edges
graph.add_edge(START, "planning_agent")
graph.add_edge("planning_agent", "supervisor_refine")
graph.add_edge("supervisor_refine", "research_decide_tools")

graph.add_conditional_edges(
    "research_decide_tools",
    route_research_tool,
    {
        "run_tool": "run_tool",
        "synthesise": "research_synthesise",
    },
)

graph.add_edge("run_tool", "research_synthesise")
graph.add_edge("research_synthesise", "supervisor_quality")

graph.add_conditional_edges(
    "supervisor_quality",
    route_quality,
    {
        "done": END,
        "retry": "supervisor_refine",   # loop back through supervisor → research
    },
)

agent = graph.compile(checkpointer=memory)