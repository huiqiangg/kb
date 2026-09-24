from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "农商行知识问答 Agent"
    log_level: str = "INFO"
    kb_platform_base_url: str = "http://127.0.0.1:8080"
    kb_project_id: str = "assets"
    kb_authorization: str = ""
    rewrite_model_url: str = "http://77.6.65.47:8099/api/model/chat/completions"
    rewrite_model_name: str = "Qwen2.5-coder-7B-Instruct"
    answer_model_url: str = ""
    answer_model_name: str = "Qwen3-32B"
    model_access_key: str = ""
    faq_similarity_threshold: float = 0.98
    retrieval_top_k: int = 12
    final_context_top_k: int = 6
    request_timeout_seconds: float = 45.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
