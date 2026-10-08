"""
Chat View Component.
Renders message history, intelligent routing badges, SQL query blocks, and source citations.
"""

from typing import Any, Dict, List
import streamlit as st


def render_route_badge(route: str):
    """Renders visual routing tag above assistant responses."""
    badge_config = {
        "sql": ("📊 SQL Database Query", "route-badge-sql"),
        "rag": ("📄 Document Knowledge Base (RAG)", "route-badge-rag"),
        "combined": ("🔀 Multi-Domain Synthesis (RAG + SQL)", "route-badge-combined"),
        "chitchat": ("💬 Conversational Persona", "route-badge-chitchat"),
        "refuse": ("🛡️ Enterprise Guardrail Refusal", "route-badge-refuse"),
    }
    label, css_class = badge_config.get(route.lower(), ("🤖 Assistant", "route-badge-chitchat"))
    st.markdown(f'<div class="route-badge {css_class}">{label}</div>', unsafe_allow_html=True)


def render_sources(sources: List[Dict[str, Any]]):
    """Renders expandable document provenance card with verified chunk citations."""
    if not sources:
        return

    with st.expander(f"📚 Verified Source Citations ({len(sources)} Chunks)", expanded=False):
        for idx, src in enumerate(sources, start=1):
            filename = src.get("filename", "Unknown")
            chunk_id = src.get("chunk_id", "N/A")
            file_type = src.get("file_type", "").upper()

            st.markdown(
                f"**{idx}. `{filename}`** ({file_type})  \n"
                f"- **Chunk ID:** `{chunk_id}`  \n"
                f"- **Provenance:** `{src.get('source_path', '')}`",
                unsafe_allow_html=True,
            )


def render_sql_drawer(sql_query: str):
    """Renders expandable SQL execution drawer with syntax highlighting."""
    if not sql_query:
        return

    with st.expander("🔍 Executed PostgreSQL Query", expanded=False):
        st.code(sql_query, language="sql")
        st.caption("🔒 Verified by AST SELECT-only security guardrails.")


def render_chat_messages(messages: List[Dict[str, Any]]):
    """Iterates through conversation history and renders message bubbles."""
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            if role == "assistant" and msg.get("route"):
                render_route_badge(msg["route"])

            st.markdown(msg["content"])

            if role == "assistant":
                if msg.get("sql_query"):
                    render_sql_drawer(msg["sql_query"])
                if msg.get("sources"):
                    render_sources(msg["sources"])
