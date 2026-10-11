"""
LLM Factory Module.
Implements the Factory Pattern to decouple LLM provider instantiation from business logic.
Supports:
- 'groq' (Groq high-speed cloud inference: llama-3.3-70b-versatile)
- 'ollama' (100% local, offline, privacy-first)
- 'openai' (OpenAI API models: gpt-4o, gpt-4o-mini)
- 'openrouter' (OpenRouter unified API gateway)
"""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for instantiating LangChain chat models and embedding providers."""

    @staticmethod
    def get_chat_model(temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
        """
        Creates and returns a BaseChatModel instance based on application configuration.

        Args:
            temperature: Sampling temperature for generation (default: 0.0 for deterministic answers).
            **kwargs: Extra parameters to pass to the model constructor.

        Returns:
            An instantiated LangChain BaseChatModel.
        """
        provider = (settings.llm.LLM_PROVIDER or "ollama").strip().lower()

        if provider == "ollama":
            logger.info(
                f"[LLMFactory] Initializing Local Ollama chat model: {settings.llm.LLM_MODEL} "
                f"at {settings.llm.OLLAMA_BASE_URL}"
            )
            return ChatOllama(
                model=settings.llm.LLM_MODEL,
                base_url=settings.llm.OLLAMA_BASE_URL,
                temperature=temperature,
                **kwargs,
            )

        elif provider in ("openai", "groq", "openrouter"):
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                logger.error(
                    "langchain-openai is required for cloud providers. "
                    "Install with: pip install langchain-openai"
                )
                raise RuntimeError(
                    f"Provider '{provider}' requested, but langchain-openai is not installed."
                ) from e

            # Retrieve API key, base_url, and model specific to provider
            if provider == "groq":
                api_key = settings.llm.GROQ_API_KEY or settings.llm.OPENAI_API_KEY
                base_url = settings.llm.OPENAI_BASE_URL or "https://api.groq.com/openai/v1"
                model_name = settings.llm.GROQ_MODEL or "llama-3.3-70b-versatile"
            elif provider == "openrouter":
                api_key = settings.llm.OPENROUTER_API_KEY or settings.llm.OPENAI_API_KEY
                base_url = settings.llm.OPENAI_BASE_URL or "https://openrouter.ai/api/v1"
                model_name = settings.llm.OPENROUTER_MODEL or "meta-llama/llama-3.3-70b-instruct"
            else:  # openai
                api_key = settings.llm.OPENAI_API_KEY
                base_url = settings.llm.OPENAI_BASE_URL
                model_name = settings.llm.OPENAI_MODEL or "gpt-4o-mini"

            if not api_key:
                logger.warning(
                    f"[LLMFactory] API key is not configured for provider '{provider}'. "
                    "Falling back to local Ollama."
                )
                return ChatOllama(
                    model=settings.llm.LLM_MODEL,
                    base_url=settings.llm.OLLAMA_BASE_URL,
                    temperature=temperature,
                    **kwargs,
                )

            logger.info(
                f"[LLMFactory] Initializing {provider.upper()} chat model: {model_name} "
                f"(base_url={base_url or 'default'})"
            )
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                **kwargs,
            )

        else:
            logger.warning(
                f"[LLMFactory] Unknown provider '{provider}'. Falling back to local Ollama."
            )
            return ChatOllama(
                model=settings.llm.LLM_MODEL,
                base_url=settings.llm.OLLAMA_BASE_URL,
                temperature=temperature,
                **kwargs,
            )


# Singleton instance of LLMFactory
llm_factory = LLMFactory()
