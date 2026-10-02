from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XCRS_", env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://xcrs:xcrs@localhost:5432/xcrs"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_batch_size: int = 32
    embedding_timeout_s: float = 120.0

    # Explanation LLM: any OpenAI-compatible chat endpoint (Ollama locally).
    llm_enabled: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:9b"
    llm_timeout_s: float = 180.0
    llm_disable_thinking: bool = True
    llm_api_key: str | None = None  # only for hosted endpoints; Ollama needs none
    # Cap on generated tokens. A v2 explanation needs ~100–200. Insurance against a known failure mode of small
    # models in JSON mode (endless whitespace), which would hold the worker until the timeout.
    llm_max_tokens: int = 500
    # Skill matching (ADR-0030) uses the same LLM; a new phrase takes ~2 s on a GPU, ~5 s or more on a CPU VM.
    match_llm_timeout_s: float = 60.0
    youtube_api_key: str | None = None  # XCRS_YOUTUBE_API_KEY; the YouTube adapter is off without it (ADR-0033)
    # Abuse protection (ADR-0035): per-client-IP token buckets; at most this many new phrases per request go
    # to the LLM, the rest use the embedding fallback.
    rate_limits_enabled: bool = True
    rate_heavy_per_minute: float = 10
    rate_heavy_burst: int = 5
    rate_light_per_minute: float = 60
    rate_light_burst: int = 20
    trusted_proxies: str = "127.0.0.1/32,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
    match_llm_max_new: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
