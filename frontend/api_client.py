"""
API Client Module for Frontend.
Handles HTTP communication with the FastAPI backend endpoints.
"""

import logging
from typing import Any, Dict, List, Optional
import httpx
from frontend.config import BACKEND_URL

logger = logging.getLogger(__name__)


class APIClient:
    """Communicates with the local GenAI Assistant FastAPI server."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        """Checks backend health status and active model."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {"status": "offline", "error": str(e)}

    def send_chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends user message to POST /chat with intelligent multi-domain routing.

        Returns:
            Dictionary with response, route, sources, sql_query, session_id.
        """
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(f"{self.base_url}/chat", json=payload)
                if resp.status_code == 200:
                    return {"success": True, "data": resp.json()}
                return {
                    "success": False,
                    "error": f"Server error ({resp.status_code}): {resp.text}",
                }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "Could not connect to FastAPI backend. Ensure the server is running on port 8000.",
            }
        except httpx.ReadTimeout:
            return {
                "success": False,
                "error": "Request timed out. Local LLM inference took longer than expected.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_documents(self) -> Dict[str, Any]:
        """Fetches list of indexed documents and collection statistics."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/documents")
                if resp.status_code == 200:
                    return {"success": True, "data": resp.json()}
                return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ingest_documents(self) -> Dict[str, Any]:
        """Triggers document re-indexing into Qdrant."""
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(f"{self.base_url}/documents/ingest")
                if resp.status_code == 200:
                    return {"success": True, "data": resp.json()}
                return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Singleton instance
api_client = APIClient()
