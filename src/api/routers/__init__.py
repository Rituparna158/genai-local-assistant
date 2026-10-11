"""
API Routers Package.
Exports modular routers for Health, Chat, and Documents.
"""

from .chat import router as chat_router
from .documents import router as documents_router
from .health import router as health_router

__all__ = ["health_router", "chat_router", "documents_router"]
