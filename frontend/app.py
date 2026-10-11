"""
Main Streamlit Application Entrypoint.
Clean, streamlined Enterprise Data Assistant chat interface with persistent multi-turn history.
"""

import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from frontend.api_client import api_client
from frontend.components.chat_view import (
    render_chat_messages,
    render_route_badge,
    render_sources,
    render_sql_drawer,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Data Assistant",
    page_icon="🤖",
    layout="wide",
)

# 2. Inject Optional Clean Styles
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# 3. Session State Initialization (persists across chat turns in the same browser session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Minimal Clean Sidebar
with st.sidebar:
    st.subheader("⚙️ System Status")
    health = api_client.get_health()
    if health.get("status") == "healthy":
        st.success("🟢 System Online")
    else:
        st.error("🔴 Backend Disconnected")

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# 5. Main Chat Interface Header
st.title("🤖 Enterprise Data Assistant")

# 6. Render Persistent Conversation History
render_chat_messages(st.session_state.messages)

# 7. React to Incoming User Message (Official Streamlit Chat Pattern)
if user_input := st.chat_input("Ask a question about policies, orders, customers, or sales..."):
    # Add user message to session state and render immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Call backend and render assistant response in place
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing..."):
            resp = api_client.send_chat(user_input, session_id=st.session_state.session_id)

        if resp.get("success"):
            data = resp["data"]
            route = data.get("route")
            content = data.get("response", "No response returned.")
            sources = data.get("sources")
            sql_query = data.get("sql_query")

            if route:
                render_route_badge(route)
            st.markdown(content)
            if sql_query:
                render_sql_drawer(sql_query)
            if sources:
                render_sources(sources)

            # Persist assistant turn to message history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "route": route,
                    "sources": sources,
                    "sql_query": sql_query,
                }
            )
        else:
            err_msg = f"⚠️ Error: {resp.get('error', 'Unable to reach backend service.')}"
            st.markdown(err_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err_msg,
                    "route": "refuse",
                    "sources": None,
                    "sql_query": None,
                }
            )
