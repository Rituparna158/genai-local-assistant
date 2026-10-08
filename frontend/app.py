"""
Main Streamlit Application Entrypoint.
Enterprise GenAI Data Assistant Web Interface.
"""

import sys
from pathlib import Path
import uuid

# Ensure project root is in sys.path so Streamlit resolves 'frontend' package cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from frontend.api_client import api_client
from frontend.components import render_chat_messages, render_quick_prompts, render_sidebar

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Enterprise Styles
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# 3. Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome! I am your Enterprise Data Assistant.**\n\n"
                "I operate **100% locally** without external cloud dependencies. You can ask me:\n"
                "- 📄 **Company Policies (RAG)**: Leave entitlement, remote work security, refund terms, benefits.\n"
                "- 📊 **Database Analytics (SQL)**: Customers, products, sales revenue, inventory, and order history.\n"
                "- 🔀 **Combined Inquiries**: Blending company policy rules with live database calculations."
            ),
            "route": "chitchat",
            "sources": None,
            "sql_query": None,
        }
    ]


# 4. Handle Incoming Query
def process_query(prompt_text: str):
    """Processes user question through the backend and updates message state."""
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    # Call backend
    with st.spinner("🤖 Routing inquiry and processing locally..."):
        resp = api_client.send_chat(prompt_text, session_id=st.session_state.session_id)

    if resp.get("success"):
        data = resp["data"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("response", "No response returned."),
            "route": data.get("route"),
            "sources": data.get("sources"),
            "sql_query": data.get("sql_query"),
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ **Error:** {resp.get('error', 'Unknown backend error occurred.')}",
            "route": "refuse",
            "sources": None,
            "sql_query": None,
        })
    st.rerun()


# 5. Render Layout
# Render Sidebar
render_sidebar()

# Main Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Enterprise Data Assistant</h1>
        <p>Local Multi-Agent Orchestration powered by FastAPI, LangGraph, Qdrant, and PostgreSQL</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Test Prompt Chips
render_quick_prompts(on_prompt_select=process_query)

st.markdown("<br>", unsafe_allow_html=True)

# Render Chat History
render_chat_messages(st.session_state.messages)

# Chat Input Field
if user_input := st.chat_input("Ask a question about company policies, orders, customers, or sales..."):
    process_query(user_input)
