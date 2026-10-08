"""
Agent State Definition for LangGraph Workflow.
Defines the shared state dictionary passed across all graph nodes.
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Centralized state for the Enterprise Data Assistant LangGraph workflow.
    Tracks user input, conversational history, routing decision, retrieval contexts,
    SQL queries, execution results, citations, and the final synthesized answer.
    """
    # Conversational messages with LangGraph automatic reducer
    messages: Annotated[List[BaseMessage], add_messages]

    # Session identifier for multi-turn conversational memory
    session_id: Optional[str]

    # Current raw user inquiry
    question: str

    # Routing classification: 'rag', 'sql', 'combined', 'chitchat', 'refuse'
    route: Optional[str]

    # Document RAG artifacts
    rag_context: Optional[str]
    sources: Optional[List[Dict[str, Any]]]

    # Database SQL artifacts
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    sql_status: Optional[str]

    # Final synthesized response returned to the user
    answer: Optional[str]

    # Error details if any step fails
    error: Optional[str]
