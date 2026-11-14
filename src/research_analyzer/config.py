from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "ResearchPilot Paper Discovery App"
    app_env: str = "local"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://researchpilot:researchpilot@localhost:5432/researchpilot"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "research_paper_chunks"
    embedding_model: str = 
    embedding_dim: int = 384
    chunk_size_words: int = 450
    chunk_overlap_words: int = 80
    max_upload_mb: int = 30
    upload_dir: str = "./data/uploads"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings():
    return Settings()

settings = get_settings()
