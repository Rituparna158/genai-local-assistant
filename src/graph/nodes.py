"""
LangGraph Nodes Module.
Defines individual node functions for routing, document RAG, database SQL,
combined synthesis, conversational chitchat, and policy refusal.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.api.ollama_service import ollama_service
from src.graph.router import router
from src.graph.state import AgentState
from src.prompts import (
    CHITCHAT_PROMPT,
    COMBINED_SYNTHESIS_PROMPT,
    CONTEXTUALIZE_PROMPT,
    GUARDRAIL_REFUSAL_MESSAGE,
)
from src.rag.rag_service import rag_service
from src.sql_agent.sql_service import sql_service

logger = logging.getLogger(__name__)


async def contextualize_node(state: AgentState) -> dict[str, Any]:
    """Resolves follow-up references using conversational history stored in state['messages']."""
    messages = state.get("messages", [])
    raw_question = state["question"]

    # First turn: no context needed
    if len(messages) <= 1:
        return {"question": raw_question}

    # Format previous turns (excluding the current user message)
    history_turns = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_turns.append(f"{role}: {msg.content}")

    chat_history = "\n".join(history_turns[-6:])  # Keep last 3 turns for context

    try:
        prompt = CONTEXTUALIZE_PROMPT.invoke(
            {
                "chat_history": chat_history,
                "question": raw_question,
            }
        )
        resp = await ollama_service.llm.ainvoke(prompt)
        reformulated = str(resp.content).strip()

        if reformulated and len(reformulated) > 2:
            logger.info(f"[GRAPH] Contextualized follow-up: '{raw_question}' -> '{reformulated}'")
            return {"question": reformulated}
    except Exception as e:
        logger.warning(f"Contextualization failed: {e}. Falling back to raw question.")

    return {"question": raw_question}


async def router_node(state: AgentState) -> dict[str, Any]:
    """Inspects user inquiry and decides which processing pathway to trigger."""
    question = state["question"]
    route = await router.classify_route(question)
    logger.info(f"[GRAPH] Routing question '{question}' to node '{route}'")
    return {"route": route}


async def rag_node(state: AgentState) -> dict[str, Any]:
    """Retrieves policy documents from Qdrant and generates a grounded response."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing RAG node for: '{question}'")
    try:
        rag_result = await rag_service.answer_question(question, top_k=4)
        answer = rag_result["answer"]
        return {
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "rag_context": rag_result.get("context", ""),
            "messages": [AIMessage(content=answer)],
        }
    except Exception as e:
        logger.error(f"[GRAPH] RAG execution failed: {e}", exc_info=True)
        fallback = (
            "An error occurred while retrieving policy documents. "
            "Please ensure documents are indexed and vector storage is reachable."
        )
        return {
            "answer": fallback,
            "sources": [],
            "rag_context": "",
            "error": str(e),
            "messages": [AIMessage(content=fallback)],
        }


async def sql_node(state: AgentState) -> dict[str, Any]:
    """Generates, validates, and executes SQL against the PostgreSQL database."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing SQL node for: '{question}'")
    try:
        sql_result = await sql_service.execute_nl_query(question)
        answer = sql_result["answer"]
        return {
            "answer": answer,
            "sql_query": sql_result.get("generated_sql"),
            "sql_result": sql_result.get("results"),
            "sql_status": sql_result.get("status"),
            "messages": [AIMessage(content=answer)],
        }
    except Exception as e:
        logger.error(f"[GRAPH] SQL execution failed: {e}", exc_info=True)
        fallback = (
            "An error occurred while querying the database. "
            "Please verify database connection and schema availability."
        )
        return {
            "answer": fallback,
            "sql_query": None,
            "sql_result": [],
            "sql_status": "error",
            "error": str(e),
            "messages": [AIMessage(content=fallback)],
        }


async def combined_node(state: AgentState) -> dict[str, Any]:
    """Executes both document retrieval and database queries concurrently, then synthesizes a unified answer."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing Combined node concurrently for: '{question}'")

    # Run Document RAG and SQL Agent in parallel with exception shielding
    rag_task = rag_service.answer_question(question, top_k=4)
    sql_task = sql_service.execute_nl_query(question)
    raw_results = await asyncio.gather(rag_task, sql_task, return_exceptions=True)

    rag_err = raw_results[0] if isinstance(raw_results[0], Exception) else None
    sql_err = raw_results[1] if isinstance(raw_results[1], Exception) else None

    if rag_err:
        logger.error(f"[GRAPH] RAG branch failed in combined node: {rag_err}")
        rag_result: dict[str, Any] = {
            "answer": "Policy context temporarily unavailable.",
            "context": "Policy context temporarily unavailable due to a retrieval error.",
            "sources": [],
        }
    else:
        rag_result = raw_results[0]  # type: ignore

    if sql_err:
        logger.error(f"[GRAPH] SQL branch failed in combined node: {sql_err}")
        sql_result: dict[str, Any] = {
            "answer": "Database query temporarily unavailable.",
            "generated_sql": "N/A",
            "results": [],
            "status": "error",
        }
    else:
        sql_result = raw_results[1]  # type: ignore

    # Step 3: Synthesis of both domains
    try:
        prompt = COMBINED_SYNTHESIS_PROMPT.invoke(
            {
                "question": question,
                "rag_answer": rag_result.get("answer", ""),
                "sql_answer": sql_result.get("answer", ""),
                "sql_query": sql_result.get("generated_sql", "N/A"),
            }
        )
        llm_resp = await ollama_service.llm.ainvoke(prompt)
        answer = str(llm_resp.content).strip()
    except Exception as e:
        logger.error(f"[GRAPH] Synthesis failed in combined node: {e}", exc_info=True)
        answer = f"{rag_result.get('answer', '')}\n\n{sql_result.get('answer', '')}".strip()

    return {
        "answer": answer,
        "sources": rag_result.get("sources", []),
        "rag_context": rag_result.get("context", ""),
        "sql_query": sql_result.get("generated_sql"),
        "sql_result": sql_result.get("results"),
        "sql_status": sql_result.get("status"),
        "messages": [AIMessage(content=answer)],
    }


async def chitchat_node(state: AgentState) -> dict[str, Any]:
    """Generates friendly conversational response for greetings and pleasantries."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing Chitchat node for: '{question}'")

    prompt = CHITCHAT_PROMPT.invoke({"question": question})
    llm_resp = await ollama_service.llm.ainvoke(prompt)
    answer = str(llm_resp.content).strip()

    return {
        "answer": answer,
        "sources": None,
        "messages": [AIMessage(content=answer)],
    }


async def refuse_node(state: AgentState) -> dict[str, Any]:
    """Politely declines out-of-scope or unsafe requests."""
    logger.info(f"[GRAPH] Executing Refusal guardrail node for: '{state['question']}'")
    answer = GUARDRAIL_REFUSAL_MESSAGE
    return {
        "answer": answer,
        "sources": None,
        "messages": [AIMessage(content=answer)],
    }
