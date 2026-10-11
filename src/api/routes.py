"""
Legacy routes module re-exporting modular routers from src.api.routers.
Maintained for backwards compatibility.
"""

from src.api.routers import chat_router, documents_router, health_router

__all__ = ["health_router", "chat_router", "documents_router"]
