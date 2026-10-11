"""
Health Router.
Provides service health and backing dependency diagnostics.
"""

from fastapi import APIRouter, status

from src.api.schemas import HealthResponse
from src.config import settings
from src.rag.vector_store import vector_store_manager
from src.sql_agent.db_connector import db_connector

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns current service status, backing dependency connectivity, active environment, and configured LLM.",
)
async def health_check() -> HealthResponse:
    """Returns the operational status of the service and connected storage systems."""
    dependencies: dict[str, str] = {}

    # Check Database connection
    try:
        db_connector.execute_query("SELECT 1;")
        dependencies["database"] = "connected"
    except Exception:
        dependencies["database"] = "disconnected"

    # Check Vector store (Qdrant)
    try:
        vector_store_manager.client.get_collections()
        dependencies["vector_store"] = "connected"
    except Exception:
        dependencies["vector_store"] = "disconnected"

    overall_status = (
        "healthy" if all(v == "connected" for v in dependencies.values()) else "degraded"
    )

    active_model = settings.llm.LLM_MODEL

    return HealthResponse(
        status=overall_status,
        app_env=settings.app.APP_ENV,
        model=active_model,
        dependencies=dependencies,
    )
