# Research Paper Discovery (Group 001-015 / VM07)

Agentic RAG app: LLM query refinement -> arXiv / Semantic Scholar search -> local vector index (Qdrant) -> ranked results -> structured paper comparison. PDF upload supported.

## Setup (WSL)
1. Keep the project in the Linux filesystem (e.g. `~/paper-discovery`), not under `/mnt/c`.
2. Copy both certificates from Windows into `certs/` (WSL sees Windows Downloads at `/mnt/c/Users/<WindowsUser>/Downloads`):
   `cp /mnt/c/Users/<WindowsUser>/Downloads/vm0[27]-ca.crt certs/`
3. `cp .env.example .env`, then `nano .env` and fill `GROUP_API_KEY` and `EMBED_API_KEY`. Never commit or share them.
   The backend finds `.env` in the project root automatically; a shell `export GROUP_API_KEY=...` overrides it.
3. Python 3.11+:
   ```
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -m backend.check_llm                            # tests the chat endpoint
   python -m backend.check_embed                          # tests the embedding endpoint
   uvicorn backend.main:app --port 8000                   # terminal 1 (keep to ONE worker)
   streamlit run frontend/app.py                          # terminal 2
   ```

## Trace dashboard (Langfuse)
Shows how each search, question and comparison really ran: every LLM call, embedding, web search and
vector search, with timing, tokens, inputs, outputs and errors.

1. Create a project in Langfuse and copy its API keys. Self-hosted servers must be Langfuse v4.59+.
2. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env`, then restart the backend.
3. Run some queries, sign in as an admin, open the **Traces** tab and press **Load traces**.

Without the keys nothing is recorded or sent. With them, questions, prompts, paper excerpts, answers and
usernames go to the Langfuse server, and admins can read every user's queries. API keys are redacted.

## Guard rails
Applied to search, ask and compare (`backend/guardrails.py`, `backend/safe_fetch.py`):

- **Question**: blocks commands aimed at the assistant and pasted secrets, masks email addresses. This runs
  before anything is traced, stored or sent out. To ask *about* prompt injection, quote the example.
- **Limits**: per-user requests per minute and per day, simultaneous runs, and a time budget for web search.
- **Planning**: refuses requests that are not about research; keywords and years are validated.
- **Papers**: excerpts are fenced as data and instruction-like text is hidden from the model (with a note).
- **Downloads**: public http(s) addresses only, re-checked on every redirect, with size, time and page limits.
- **Answers**: citations that match no source are removed, uncited answers get a note, images and links are
  removed, secrets and system-prompt leaks are blocked. Comparisons only contain the papers you selected.

Pattern checks can be worded around; they are one layer, not a guarantee. Limits are kept in memory, so they
reset on restart and need the single worker. Settings: `GUARD_*` in `.env.example`.
Tests: `python -m unittest discover -s tests -t .`

## Layout
- `backend/guardrails.py`, `backend/safe_fetch.py` - guard rails (see above)
- `backend/tracing.py` - Langfuse spans (no-op without keys) and key redaction
- `backend/traces.py` - reads traces back for the dashboard
- `frontend/traces_tab.py` - the Traces tab (admins only)
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
