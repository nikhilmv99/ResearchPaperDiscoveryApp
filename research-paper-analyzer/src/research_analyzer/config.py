import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str
    embedding_model: str
    embedding_api_key: str
    embedding_ca_cert: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_ca_cert: str | None
    chroma_path: str

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            os.getenv("EMBEDDING_BASE_URL", "https://103.42.50.41:1035/v1"),
            os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5"),
            os.getenv("EMBEDDING_API_KEY", ""),
            os.getenv("EMBEDDING_CA_CERT", ""),
            os.getenv("LLM_BASE_URL", ""),
            os.getenv("LLM_MODEL", ""),
            os.getenv("LLM_API_KEY", ""),
            os.getenv("LLM_CA_CERT") or None,
            os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        )
