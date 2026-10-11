"""
Sidebar Component.
Displays backend status, active model, session management, and knowledge base documents.
"""

import uuid

import streamlit as st

from frontend.api_client import api_client


def render_sidebar():
    """Renders the comprehensive system control and telemetry sidebar."""
    with st.sidebar:
        st.title("⚙️ System Control")

        # 1. Backend Connectivity & Health
        st.subheader("🔌 Backend Status")
        health = api_client.get_health()
        if health.get("status") == "healthy":
            st.success("🟢 System Online")
            st.caption(f"Environment: **{health.get('app_env', 'development')}**")
        else:
            st.error(f"🔴 Offline ({health.get('error', 'Unreachable')})")
            st.caption("Ensure FastAPI is running on `http://localhost:8000`.")

        st.divider()

        # 2. Conversational Session Management
        st.subheader("💬 Session Management")
        st.caption("Active Session ID:")
        st.code(st.session_state.session_id, language=None)

        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # 3. Document Knowledge Base (Qdrant)
        st.subheader("📁 Knowledge Base")
        doc_resp = api_client.list_documents()

        if doc_resp.get("success"):
            data = doc_resp["data"]
            col1, col2 = st.columns(2)
            col1.metric("Documents", data.get("total_documents", 0))
            col2.metric("Chunks", data.get("total_chunks", 0))

            with st.expander("📄 View Indexed Documents", expanded=False):
                for doc in data.get("documents", []):
                    st.markdown(f"- **{doc['filename']}** ({doc['chunk_count']} chunks)")

            if st.button("⚡ Re-index Documents", use_container_width=True):
                with st.spinner("Ingesting and embedding documents into Qdrant..."):
                    ingest_resp = api_client.ingest_documents()
                    if ingest_resp.get("success"):
                        st.success("Re-indexed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {ingest_resp.get('error')}")
        else:
            st.warning("Could not fetch document index.")

        st.divider()

        # 4. Tech Stack & Privacy Badge
        st.caption("🔒 **Privacy Guarantee**")
        st.caption("100% Local execution with Ollama, PostgreSQL, and Qdrant. Zero cloud leakage.")
