import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() looks for a .env file starting in this folder and walking up,
# so the project-root .env is found no matter where you start the server from.
# Real shell variables (export GROUP_API_KEY=...) take priority over .env.
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _path(p: str) -> str:
    """~ is expanded; relative paths (e.g. certs/vm07-ca.crt) are relative to the project root."""
    p = Path(os.path.expanduser(p))
    return str(p if p.is_absolute() else ROOT / p)


# Chat LLM (group 001-015, VM07)
API_BASE = os.getenv("LLM_API_BASE", "https://103.42.50.41:1033/v1").rstrip("/")
API_KEY = os.getenv("GROUP_API_KEY", "")
CA_CERT = _path(os.getenv("CA_CERT_PATH", "certs/vm05-ca.crt"))
MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")

# Shared embedding API (VM02)
EMBED_API_BASE = os.getenv("EMBED_API_BASE", "https://103.42.50.41:1035/v1").rstrip("/")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_CA_CERT = _path(os.getenv("EMBED_CA_CERT_PATH", "certs/vm02-ca.crt"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-small-en-v1.5")


JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

S2_API_KEY = os.getenv("S2_API_KEY", "")
SOURCES = [s.strip() for s in os.getenv("SOURCES", "arxiv,semantic_scholar").split(",") if s.strip()]

DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Langfuse tracing (disabled unless BOTH keys are configured)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv(
    "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
).rstrip("/")

# Guardrails
GUARD_PER_MINUTE = int(os.getenv("GUARD_PER_MINUTE", "10"))
GUARD_PER_DAY = int(os.getenv("GUARD_PER_DAY", "200"))
GUARD_CONCURRENT = int(os.getenv("GUARD_CONCURRENT", "2"))
GUARD_WEB_SECONDS = int(os.getenv("GUARD_WEB_SECONDS", "90"))
GUARD_PDF_MB = int(os.getenv("GUARD_PDF_MB", "25"))