"""
LangGraph Workflow Module.
Compiles the StateGraph orchestrating router, RAG, SQL, combined, chitchat,
and refusal nodes with conversational memory checkpointing.
"""

import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    chitchat_node,
    combined_node,
    contextualize_node,
    rag_node,
    refuse_node,
    router_node,
    sql_node,
)
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_selector(state: AgentState) -> str:
    """Conditional edge selector mapping state['route'] to target node."""
    route = state.get("route", "rag")
    if route not in {"rag", "sql", "combined", "chitchat", "refuse"}:
        return "rag"
    return route


def create_agent_graph():
    """Builds and compiles the StateGraph workflow with memory checkpointing."""
    builder = StateGraph(AgentState)  # type: ignore[arg-type]

    # 1. Register nodes
    builder.add_node("contextualize", contextualize_node)
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("sql", sql_node)
    builder.add_node("combined", combined_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("refuse", refuse_node)

    # 2. Add edges: START -> contextualize -> router
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "router")

    builder.add_conditional_edges(
        "router",
        route_selector,
        {
            "rag": "rag",
            "sql": "sql",
            "combined": "combined",
            "chitchat": "chitchat",
            "refuse": "refuse",
        },
    )

    # All functional nodes terminate at END
    builder.add_edge("rag", END)
    builder.add_edge("sql", END)
    builder.add_edge("combined", END)
    builder.add_edge("chitchat", END)
    builder.add_edge("refuse", END)

    # 3. Compile with MemorySaver for session persistence
    memory = MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)
    logger.info("Compiled LangGraph Enterprise Data Assistant with MemorySaver.")
    return compiled_graph


# Singleton instance of compiled agent graph
agent_graph = create_agent_graph()


async def run_assistant(question: str, session_id: str | None = None) -> dict[str, Any]:
    """
    Executes the conversational graph workflow for an incoming question.

    Args:
        question: User query string.
        session_id: Optional session identifier for tracking conversation history.

    Returns:
        Dictionary containing answer, route, sources, sql_query, and session_id.
    """
    active_session_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": active_session_id}}

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "session_id": active_session_id,
        "route": None,
        "rag_context": None,
        "sources": None,
        "sql_query": None,
        "sql_result": None,
        "sql_status": None,
        "answer": None,
        "error": None,
    }

    logger.info(f"Invoking agent graph for session '{active_session_id}': '{question}'")
    result_state = await agent_graph.ainvoke(initial_state, config=config)

    return {
        "answer": result_state.get("answer", "No response generated."),
        "route": result_state.get("route", "unknown"),
        "sources": result_state.get("sources"),
        "sql_query": result_state.get("sql_query"),
        "sql_result": result_state.get("sql_result"),
        "session_id": active_session_id,
    }
