"""
LangGraph Nodes Module.
Defines individual node functions for routing, document RAG, database SQL,
combined synthesis, conversational chitchat, and policy refusal.
"""

import asyncio
import logging
from typing import Any, Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from src.api.ollama_service import ollama_service
from src.graph.router import router
from src.graph.state import AgentState
from src.rag.rag_service import rag_service
from src.sql_agent.sql_service import sql_service

logger = logging.getLogger(__name__)

COMBINED_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an enterprise data assistant.\n"
        "The user has asked a question that requires BOTH company policy context and database records.\n\n"
        "COMPANY POLICY CONTEXT:\n{rag_context}\n\n"
        "DATABASE ANALYSIS:\n{sql_answer}\n(SQL Query: {sql_query})\n\n"
        "TASK:\n"
        "Provide a comprehensive, unified answer that directly addresses the user's question.\n"
        "Integrate the policy rules with the specific data and metrics from the database seamlessly.\n"
        "Be factual, precise, and professional.",
    ),
    ("human", "{question}"),
])


async def router_node(state: AgentState) -> Dict[str, Any]:
    """Inspects user inquiry and decides which processing pathway to trigger."""
    question = state["question"]
    route = await router.classify_route(question)
    logger.info(f"[GRAPH] Routing question '{question}' to node '{route}'")
    return {"route": route}


async def rag_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves policy documents from Qdrant and generates a grounded response."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing RAG node for: '{question}'")
    rag_result = await rag_service.answer_question(question, top_k=3)
    answer = rag_result["answer"]

    return {
        "answer": answer,
        "sources": rag_result.get("sources", []),
        "rag_context": rag_result.get("context", ""),
        "messages": [AIMessage(content=answer)],
    }


async def sql_node(state: AgentState) -> Dict[str, Any]:
    """Generates, validates, and executes SQL against the PostgreSQL database."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing SQL node for: '{question}'")
    sql_result = await sql_service.execute_nl_query(question)
    answer = sql_result["answer"]

    return {
        "answer": answer,
        "sql_query": sql_result.get("generated_sql"),
        "sql_result": sql_result.get("results"),
        "sql_status": sql_result.get("status"),
        "messages": [AIMessage(content=answer)],
    }


async def combined_node(state: AgentState) -> Dict[str, Any]:
    """Executes both document retrieval and database queries concurrently, then synthesizes a unified answer."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing Combined node concurrently for: '{question}'")

    # Run Document RAG and SQL Agent in parallel to maximize throughput
    rag_task = rag_service.answer_question(question, top_k=3)
    sql_task = sql_service.execute_nl_query(question)
    rag_result, sql_result = await asyncio.gather(rag_task, sql_task)

    # Step 3: Synthesis of both domains
    prompt = COMBINED_SYNTHESIS_PROMPT.invoke({
        "question": question,
        "rag_context": rag_result.get("context", ""),
        "sql_answer": sql_result.get("answer", ""),
        "sql_query": sql_result.get("generated_sql", "N/A"),
    })
    llm_resp = await ollama_service.llm.ainvoke(prompt)
    answer = str(llm_resp.content).strip()

    return {
        "answer": answer,
        "sources": rag_result.get("sources", []),
        "rag_context": rag_result.get("context", ""),
        "sql_query": sql_result.get("generated_sql"),
        "sql_result": sql_result.get("results"),
        "sql_status": sql_result.get("status"),
        "messages": [AIMessage(content=answer)],
    }


async def chitchat_node(state: AgentState) -> Dict[str, Any]:
    """Generates friendly conversational response for greetings and pleasantries."""
    question = state["question"]
    logger.info(f"[GRAPH] Executing Chitchat node for: '{question}'")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the Enterprise Data Assistant.\n"
            "You assist employees with internal company policies (leave, benefits, IT security, remote work) "
            "and business database queries (customers, products, orders, sales metrics).\n"
            "Introduce yourself clearly as 'Enterprise Data Assistant' (never use placeholders like [Assistant Name]).\n"
            "Respond to the user's greeting warmly, concisely, and offer your help with company documents or data.",
        ),
        ("human", "{question}"),
    ]).invoke({"question": question})

    llm_resp = await ollama_service.llm.ainvoke(prompt)
    answer = str(llm_resp.content).strip()

    return {
        "answer": answer,
        "sources": None,
        "messages": [AIMessage(content=answer)],
    }


async def refuse_node(state: AgentState) -> Dict[str, Any]:
    """Politely declines out-of-scope or unsafe requests."""
    logger.info(f"[GRAPH] Executing Refusal guardrail node for: '{state['question']}'")
    answer = (
        "I am an enterprise data assistant designed to assist with internal company policy documents "
        "and relational database queries (customers, products, and orders). "
        "I cannot assist with requests outside this operational scope."
    )
    return {
        "answer": answer,
        "sources": None,
        "messages": [AIMessage(content=answer)],
    }
