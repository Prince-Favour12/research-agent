"""
Multi-Agent Research System
===========================
Planning Agent → Supervisor (refine) → Research Agent → Supervisor (quality check) → [retry or done]
"""

from src.agent import agent, MultiAgentState
from src.schema import RequestInput


def run(query: str, thread_id: str = "1") -> str:
    input_state: MultiAgentState = {
        "user_query": RequestInput(user_query=query),
        "intent": None,
        "structured_plan": None,
        "refined_query": None,
        "quality_feedback": None,
        "messages": [],
        "generated_answer": None,
        "retry_count": 0,
        "quality_approved": False,
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(input_state, config)
    return result["generated_answer"]


if __name__ == "__main__":
    # Save graph visualisation
    png_data = agent.get_graph().draw_mermaid_png()
    with open("multi_agent_graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved as multi_agent_graph.png\n")

    answer = run("CEO of SpaceX")
    print(f"\n{'='*60}\nFINAL ANSWER:\n{'='*60}\n{answer}")