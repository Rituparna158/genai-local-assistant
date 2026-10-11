"""
Unified Configuration Package.
Exposes modular configurations for App, LLM, Database, and Qdrant.
"""

from .app_config import AppSettings
from .db_config import DatabaseSettings
from .llm_config import LLMSettings
from .qdrant_config import QdrantSettings


class Settings:
    def __init__(self):
        self.app = AppSettings()
        self.llm = LLMSettings()
        self.db = DatabaseSettings()
        self.qdrant = QdrantSettings()


# Global settings singleton instance
settings = Settings()

__all__ = ["settings", "AppSettings", "LLMSettings", "DatabaseSettings", "QdrantSettings"]
