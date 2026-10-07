"""
Legacy routes module re-exporting modular routers from src.api.routers.
Maintained for backwards compatibility.
"""

from src.api.routers import health_router, chat_router, documents_router

__all__ = ["health_router", "chat_router", "documents_router"]
