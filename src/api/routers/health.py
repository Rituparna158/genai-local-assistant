"""
Health Router.
Provides service health and environment diagnostics.
"""

from fastapi import APIRouter, status
from src.config import settings
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns current service status, active environment name, and configured local LLM model.",
)
async def health_check() -> HealthResponse:
    """Returns the operational status of the service."""
    return HealthResponse(
        status="healthy",
        app_env=settings.app.APP_ENV,
        model=settings.llm.LLM_MODEL,
    )
