"""
Agent State Definition for LangGraph Workflow.
Defines the shared state dictionary passed across all graph nodes.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Centralized state for the Enterprise Data Assistant LangGraph workflow.
    Tracks user input, conversational history, routing decision, retrieval contexts,
    SQL queries, execution results, citations, and the final synthesized answer.
    """

    # Conversational messages with LangGraph automatic reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Session identifier for multi-turn conversational memory
    session_id: str | None

    # Current raw user inquiry
    question: str

    # Routing classification: 'rag', 'sql', 'combined', 'chitchat', 'refuse'
    route: str | None

    # Document RAG artifacts
    rag_context: str | None
    sources: list[dict[str, Any]] | None

    # Database SQL artifacts
    sql_query: str | None
    sql_result: list[dict[str, Any]] | None
    sql_status: str | None

    # Final synthesized response returned to the user
    answer: str | None

    # Error details if any step fails
    error: str | None
