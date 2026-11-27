"""FastAPI backend.  Run:  uvicorn backend.main:app --port 8000

Access rules: everything except /health and /auth/login needs a login token;
uploading PDFs and deleting vectors need the admin role.
Note: Qdrant runs in local embedded mode (no separate server), so run uvicorn
with a single worker (--workers 1) to avoid Qdrant directory lock conflicts.
"""
import logging
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import Query

from . import guardrails, tracing, traces

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import auth, chats, compare, config, graph, ingest, qa, store

log = logging.getLogger(__name__)
MAX_UPLOAD = config.GUARD_PDF_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.init()
    try:
        yield
    finally:
        tracing.shutdown()

app = FastAPI(title="Research Paper Discovery", lifespan=lifespan)
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:  # development: the Flutter web dev server runs on a random localhost port
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )


class LoginReq(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


class SearchReq(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    use_web: bool = True


class VectorSearchReq(BaseModel):
    query: str | None = Field(default=None, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=200)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    paper_ids: list[str] = Field(default=[], max_length=50)
    kind: Literal["abstract", "fulltext"] | None = None


class CompareReq(BaseModel):
    paper_ids: list[str] = Field(max_length=3)
    focus: str | None = Field(default=None, max_length=200)


class ChatCreate(BaseModel):
    title: str = Field(default="New chat", max_length=80)


class ChatRename(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ChatAsk(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    paper_ids: list[str] = Field(default=[], max_length=50)  # empty = all uploaded papers


def _fail(what: str, e: Exception) -> HTTPException:
    log.exception(what)
    return HTTPException(502, f"{what} ({type(e).__name__}). See the backend log.")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/auth/login")
def login(req: LoginReq, request: Request):
    user = auth.authenticate(req.username, req.password, request.client.host if request.client else "")
    if not user:
        raise HTTPException(401, "Wrong username or password")
    return {"access_token": auth.make_token(user["username"]), **user}


@app.get("/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return user

def _record_block(endpoint: str, user: dict, e: guardrails.Blocked) -> None:
    log.warning("guardrail blocked endpoint=%s user=%s rule=%s", endpoint, user["username"], e.rule)
    if e.rule == "scope":  # raised inside the trace, which already records it
        return
    with tracing.trace("blocked", user=user["username"], input={"endpoint": endpoint, "rule": e.rule}) as obs:
        obs.update(level="WARNING", status_message=e.rule)  # rule only, never the text (it may hold a secret)

@app.post("/search")
def search(req: SearchReq, user: dict = Depends(auth.current_user)):
    try:
        question = guardrails.check_input(req.question)

        with guardrails.slot(user["username"]):

            with tracing.trace(
                "search",
                user=user["username"],
                input={
                    "question": question,
                    "use_web": req.use_web,
                },
            ) as obs:

                result = graph.run_search(question, req.use_web)

                obs.update(output={
                    "plan": result.get("plan", {}),
                    "result_count": len(result.get("results", [])),
                    "paper_ids": [
                        p["paper_id"]
                        for p in result.get("results", [])
                    ],
                })

                return result

    except guardrails.Blocked as e:
        _record_block("search", user, e)
        raise HTTPException(e.status, e.message)

    except Exception as e:
        raise _fail("Search failed", e)


@app.get("/papers")
def papers(source: str | None = None, user: dict = Depends(auth.current_user)):
    rows = [p for p in store.get_papers() if not source or p["source"] == source]
    return [{k: p[k] for k in ("paper_id", "title", "year", "source")} for p in rows]


@app.post("/upload")
def upload(file: UploadFile = File(...), admin: dict = Depends(auth.require_admin)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(
            400,
            f"PDF is larger than {config.GUARD_PDF_MB} MB",
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")
    try:
        with tracing.trace("upload", user=admin["username"],
                           input={"filename": file.filename, "bytes": len(data)}) as obs:
            p = ingest.ingest_upload(file.filename, data)
            obs.update(output={"paper_id": p["paper_id"]})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise _fail("Upload failed", e)
    return {"paper_id": p["paper_id"], "title": p["title"]}


# ----------------------------------------------------------------------------- Qdrant vector routes
@app.get("/vectors/stats")
def vector_stats(admin: dict = Depends(auth.require_admin)):
    try:
        return store.collection_info()
    except Exception as e:
        raise _fail("Failed to read Qdrant stats", e)


@app.post("/vectors/search")
def vector_search(req: VectorSearchReq, admin: dict = Depends(auth.require_admin)):
    try:
        hits = store.search_or_list_chunks(
            query=req.query,
            k=req.top_k,
            year_from=req.year_from,
            paper_ids=req.paper_ids or None,
            kind=req.kind,
        )
        return {"results": hits}
    except Exception as e:
        raise _fail("Vector search failed", e)


@app.get("/vectors/papers/{paper_id}/chunks")
def paper_chunk_counts(paper_id: str, user: dict = Depends(auth.current_user)):
    if not store.has_paper(paper_id):
        raise HTTPException(404, "Paper not found")
    try:
        abstract_cnt = store.count_chunks(paper_id, kind="abstract")
        fulltext_cnt = store.count_chunks(paper_id, kind="fulltext")
        return {
            "paper_id": paper_id,
            "abstract": abstract_cnt,
            "fulltext": fulltext_cnt,
            "total": abstract_cnt + fulltext_cnt,
        }
    except Exception as e:
        raise _fail("Failed to count chunks", e)


@app.delete("/vectors/papers/{paper_id}")
def delete_paper_vectors(paper_id: str, admin: dict = Depends(auth.require_admin)):
    if not store.has_paper(paper_id):
        raise HTTPException(404, "Paper not found")
    try:
        store.delete_paper_vectors(paper_id)
        return {"ok": True, "paper_id": paper_id}
    except Exception as e:
        raise _fail("Failed to delete vectors", e)


def _own_chat(user: dict, chat_id: str) -> dict:
    chat = chats.get_chat(user["username"], chat_id)  # None for missing AND for other users' chats
    if not chat:
        raise HTTPException(404, "Chat not found")
    return chat


@app.get("/chats")
def list_chats(user: dict = Depends(auth.current_user)):
    return chats.list_chats(user["username"])


@app.post("/chats")
def create_chat(req: ChatCreate, user: dict = Depends(auth.current_user)):
    try:
        return chats.create_chat(user["username"], req.title)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str, user: dict = Depends(auth.current_user)):
    return _own_chat(user, chat_id)


@app.patch("/chats/{chat_id}")
def rename_chat(chat_id: str, req: ChatRename, user: dict = Depends(auth.current_user)):
    if not chats.rename_chat(user["username"], chat_id, req.title.strip()):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user: dict = Depends(auth.current_user)):
    if not chats.delete_chat(user["username"], chat_id):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.post("/chats/{chat_id}/ask")
def chat_ask(
    chat_id: str,
    req: ChatAsk,
    user: dict = Depends(auth.current_user),
):
    chat = _own_chat(user, chat_id)

    try:
        question = guardrails.check_input(req.question)
        history = chat["messages"][-4:]

        with guardrails.slot(user["username"]):

            with tracing.trace(
                "ask",
                user=user["username"],
                session=chat_id,
                input={
                    "question": question,
                    "paper_ids": req.paper_ids,
                    "history_messages": len(history),
                },
            ) as obs:

                res = qa.answer_question(
                    question,
                    req.paper_ids,
                    history,
                )

                chats.add_exchange(
                    user["username"],
                    chat_id,
                    question,
                    res["answer"],
                    res["sources"],
                )

                obs.update(output={
                    "answer": res["answer"][:1000],
                    "source_count": len(res["sources"]),
                    "warnings": res.get("warnings", []),
                    "masked_papers": len(res.get("masked_papers", [])),
                })

                return res

    except guardrails.Blocked as e:
        _record_block("ask", user, e)
        raise HTTPException(e.status, e.message)

    except Exception as e:
        raise _fail("Question failed", e)

@app.post("/compare")
def compare_endpoint(
    req: CompareReq,
    user: dict = Depends(auth.current_user),
):
    try:
        focus = (
            guardrails.check_input(req.focus)
            if req.focus and req.focus.strip()
            else None
        )

        with guardrails.slot(user["username"]):

            with tracing.trace(
                "compare",
                user=user["username"],
                input={
                    "question": "Compare selected papers: " + ", ".join(req.paper_ids),
                    "paper_ids": req.paper_ids,
                    "focus": focus,
                },
            ) as obs:

                result = compare.compare_papers(
                    req.paper_ids,
                    focus,
                )

                obs.update(output={
                    "paper_count": len(result.get("papers", [])),
                    "paper_ids": [
                        p.get("id")
                        for p in result.get("papers", [])
                    ],
                    "summary": result.get("summary", "")[:500],
                    "warnings": result.get("warnings", []),
                    "error": result.get("error"),
                })

                if result.get("error"):
                    obs.update(
                        level="ERROR",
                        status_message=str(result["error"])[:300],
                    )

                return result

    except guardrails.Blocked as e:
        _record_block("compare", user, e)
        raise HTTPException(e.status, e.message)

    except ValueError as e:
        raise HTTPException(400, str(e))

    except Exception as e:
        raise _fail("Comparison failed", e)

@app.get("/traces")
def list_traces(
    hours: int = Query(default=24, ge=1, le=720),
    admin: dict = Depends(auth.require_admin),
):
    try:
        return traces.list_traces(hours, limit=500)
    except Exception as e:
        raise _fail("Failed to load traces", e)


@app.get("/traces/{trace_id}")
def get_trace(
    trace_id: str,
    admin: dict = Depends(auth.require_admin),
):
    try:
        result = traces.get_trace(trace_id)

        if result is None:
            raise HTTPException(404, "Trace not found")

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise _fail("Failed to load trace", e)