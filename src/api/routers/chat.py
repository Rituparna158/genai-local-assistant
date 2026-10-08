"""
Chat Router.
Handles conversational interactions with intelligent routing across
Document RAG, Database SQL, Combined analysis, Chitchat, and Guardrail refusal.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from src.config import settings
from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.graph import run_assistant

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Assistant (Multi-Domain Intelligent Routing)",
    description=(
        "Processes user inquiries through a LangGraph workflow that intelligently routes "
        "between Document RAG, Database SQL, Combined multi-source analysis, chitchat, and safety guardrails."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Processes user query and generates an AI response via the LangGraph orchestrator."""
    try:
        agent_result = await run_assistant(
            question=request.message,
            session_id=request.session_id,
        )

        citations = None
        if agent_result.get("sources"):
            citations = [
                SourceCitation(
                    filename=src["filename"],
                    chunk_id=src["chunk_id"],
                    file_type=src["file_type"],
                    source_path=src["source_path"],
                )
                for src in agent_result["sources"]
            ]

        return ChatResponse(
            response=agent_result["answer"],
            model=settings.llm.LLM_MODEL,
            session_id=agent_result["session_id"],
            route=agent_result.get("route"),
            sources=citations,
            sql_query=agent_result.get("sql_query"),
        )
    except Exception as e:
        logger.error(f"Error generating chat response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate assistant response: {str(e)}",
        )
