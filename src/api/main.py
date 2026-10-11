"""
FastAPI Application Entrypoint.
Initializes the FastAPI application, mounts modular routers, and configures Swagger metadata.
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routers import chat_router, documents_router, health_router

app = FastAPI(
    title="Local GenAI Data Assistant API",
    description=(
        "A fully local GenAI Assistant that can chat with documents and SQL data. "
        "Runs entirely on local infrastructure with zero business data sent to external cloud services."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register modular domain routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects the root URL to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")
