import streamlit as st

from main import run

st.set_page_config(page_title="ChatGPT-like Chat", page_icon="💬", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI assistant. Ask me anything."
        }
    ]


def get_ai_response(user_text: str) -> str:
    """AI response logic. Replace this with an actual API call to your AI model."""
    # Placeholder response logic. Replace this with a real AI API call if available.
    return run(user_text)

st.title("ChatGPT-like Chat UI")
st.write("Send a message and see the AI assistant reply below.")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Type your message", height=120, placeholder="Ask a question...")
    submitted = st.form_submit_button("Send")
    if submitted:
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            ai_reply = get_ai_response(user_input.strip())
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        else:
            st.warning("Please enter a message before sending.")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        if hasattr(st, "chat_message"):
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            st.markdown(f"**You:** {msg['content']}")
    else:
        if hasattr(st, "chat_message"):
            with st.chat_message("assistant"):
                st.write(msg["content"])
        else:
            st.markdown(f"**AI:** {msg['content']}")

