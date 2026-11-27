# Research Paper Discovery (Group 001-015 / VM07)

Agentic RAG app: LLM query refinement -> arXiv / Semantic Scholar search -> local vector index (Qdrant) -> ranked results -> structured paper comparison. PDF upload supported.

## Setup (WSL)
1. Keep the project in the Linux filesystem (e.g. `~/paper-discovery`), not under `/mnt/c`.
2. Copy both certificates from Windows into `certs/` (WSL sees Windows Downloads at `/mnt/c/Users/<WindowsUser>/Downloads`):
   `cp /mnt/c/Users/<WindowsUser>/Downloads/vm0[27]-ca.crt certs/`
3. `cp .env.example .env`, then `nano .env` and fill `GROUP_API_KEY` and `EMBED_API_KEY`. Never commit or share them.
   The backend finds `.env` in the project root automatically; a shell `export GROUP_API_KEY=...` overrides it.
3. Python 3.10+:
   ```
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -m backend.check_llm                            # tests the chat endpoint
   python -m backend.check_embed                          # tests the embedding endpoint
   uvicorn backend.main:app --port 8000                   # terminal 1 (keep to ONE worker)
   streamlit run frontend/app.py                          # terminal 2
   ```

## Layout
- `backend/llm.py` - group endpoint client: 1 request at a time, <60/min, max 1024 output tokens, retry 2/4/8s on 429/503
- `backend/graph.py` - LangGraph: plan -> web (optional) -> local vector search
- `backend/sources.py` - arXiv + Semantic Scholar (Web Search Agent)
- `backend/store.py` - SQLite metadata + Qdrant vectors (embedded, no server)
- `backend/embeddings.py` - shared embedding API (bge-small-en-v1.5, 384-d, 16 texts per request)
- `backend/ingest.py` - PDF -> text -> chunks (PyMuPDF)
- `backend/compare.py` - evidence retrieval per aspect + one LLM call (kept inside the 8192-token limit)
- `frontend/app.py` - Streamlit UI

## Notes
- Only one active request per key: if teammates share the key, one gets 429 while another runs.
- Papers found on the web are indexed by abstract; Compare downloads the PDF and indexes full text on demand.
- Next steps: GROBID for section-aware chunks, hybrid (BM25 + vector) ranking, Docker Compose.
