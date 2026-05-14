import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# 1. Local Knowledge Base (No LLM needed)
FAQ_DATA = {
    "password": "To reset your password, click 'Forgot Password' on the login page and check your email.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "refund": "You can request a refund within 30 days of purchase through your account dashboard.",
    "contact": "You can reach our human support team at support@example.com or call 1-800-555-0199."
}

# 2. Define the Graph State
class ChatbotState(TypedDict):
    user_message: str
    detected_keywords: List[str]
    raw_response: str

# 3. Define the Nodes
def intent_classifier_node(state: ChatbotState) -> dict:
    """Scans the user message for keywords in our database."""
    print("\n[Node 1: Intent Classifier] Scanning user input...")
    time.sleep(0.8) # Simulate processing latency
    
    text = state["user_message"].lower()
    found = [kw for kw in FAQ_DATA.keys() if kw in text]
    
    return {"detected_keywords": found}

def check_routing(state: ChatbotState) -> str:
    """Conditional edge logic determining where to route next."""
    print("[Edge Router] Evaluating confidence...")
    if state["detected_keywords"]:
        return "found"
    return "not_found"

def knowledge_retrieval_node(state: ChatbotState) -> dict:
    """Fetches the matching text from our local dictionary."""
    print("[Node 2A: Knowledge Base] Fetching answer...")
    time.sleep(1.0) # Simulate database fetch latency
    
    # Grab the answer for the first matched keyword
    matched_keyword = state["detected_keywords"][0]
    answer = FAQ_DATA[matched_keyword]
    
    return {"raw_response": answer}

def fallback_node(state: ChatbotState) -> dict:
    """Handles cases where no keywords matched."""
    print("[Node 2B: Fallback Agent] Generating generic help message...")
    time.sleep(0.5)
    
    fallback_text = "I'm sorry, I couldn't find an exact match for that. Type 'contact' to get our support details."
    return {"raw_response": fallback_text}

# 4. Build and Compile the Graph
builder = StateGraph(ChatbotState)

builder.add_node("classifier", intent_classifier_node)
builder.add_node("retriever", knowledge_retrieval_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    check_routing,
    {
        "found": "retriever",
        "not_found": "fallback"
    }
)

builder.add_edge("retriever", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# 5. Streaming Execution
user_query = "Hey, I need help with my password reset and shipping info"
print(f"User Sent: '{user_query}'")
print("-" * 50)

# We use stream_mode="updates" to see nodes complete one-by-one
for chunk in graph.stream({"user_message": user_query}, stream_mode="updates"):
    for node_name, state_update in chunk.items():
        print(f"\n📢 STREAM EVENT -> Node '{node_name}' finished executing.")
        print(f"   State Changes Emitted: {state_update}")
        
        # If the final text answer is ready in the stream, simulate token-by-token text output
        if "raw_response" in state_update:
            print("\n🤖 Bot Response Stream: ", end="", flush=True)
            for word in state_update["raw_response"].split():
                print(word + " ", end="", flush=True)
                time.sleep(0.15) # Simulates text generation typing speed
            print() # New line after text stream finishes

print("-" * 50)
print("Graph Finished Processing.")
