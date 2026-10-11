"""Multi-provider LLM and Embeddings configuration (Groq, Ollama, OpenAI, OpenRouter)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    # Provider selection: 'groq', 'ollama' (default local), 'openai', 'openrouter'
    LLM_PROVIDER: str = "ollama"

    # Ollama settings (Local / Offline)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Groq settings (High-Speed Cloud Inference)
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # OpenRouter settings
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct"

    # OpenAI settings
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
