"""
Quick Prompts Component.
Provides 1-click prompt chip buttons for demonstrating different routing pathways.
"""

from typing import Callable
import streamlit as st
from frontend.config import SAMPLE_QUERIES


def render_quick_prompts(on_prompt_select: Callable[[str], None]):
    """Renders categorized sample query buttons for instant testing and demos."""
    st.markdown("##### 💡 Quick Test Scenarios")

    tabs = st.tabs(list(SAMPLE_QUERIES.keys()))

    for tab, (category, queries) in zip(tabs, SAMPLE_QUERIES.items()):
        with tab:
            cols = st.columns(len(queries))
            for col, query in zip(cols, queries):
                with col:
                    if st.button(query, key=f"btn_{query[:25]}", use_container_width=True):
                        on_prompt_select(query)
