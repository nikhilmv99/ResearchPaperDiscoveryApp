"""Storage: SQLite for paper metadata, Qdrant (local mode, no server) for vectors."""
import sqlite3
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny,
                                  MatchValue, PointStruct, Range, VectorParams)

from . import config, embeddings, tracing

COLL = "chunks"
_qd = QdrantClient(path=str(config.DATA_DIR / "qdrant"))
if not _qd.collection_exists(COLL):
    _qd.create_collection(COLL, vectors_config=VectorParams(size=embeddings.DIM, distance=Distance.COSINE))

_DB = str(config.DATA_DIR / "papers.db")
_COLS = ["paper_id", "title", "authors", "year", "abstract", "url", "pdf_url", "source", "citations"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, abstract TEXT,
        url TEXT, pdf_url TEXT, source TEXT, citations INTEGER)""")
    return c


def has_paper(pid: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM papers WHERE paper_id=?", (pid,)).fetchone() is not None


def upsert_paper(p: dict) -> None:
    with _conn() as c:
        c.execute(f"INSERT OR REPLACE INTO papers VALUES ({','.join('?' * len(_COLS))})",
                  [p.get(k) for k in _COLS])


def get_papers(ids: list[str] | None = None) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM papers").fetchall()
    papers = [dict(r) for r in rows]
    return [p for p in papers if p["paper_id"] in ids] if ids is not None else papers


def add_chunks(items: list[dict]) -> None:
    """items: {paper_id, year, kind ('abstract'|'fulltext'), idx, text}"""
    with tracing.span("index chunks", input={"chunks": len(items)}):
        for i in range(0, len(items), 64):
            batch = items[i:i + 64]
            vecs = embeddings.embed([b["text"] for b in batch])
            pts = [PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{b['paper_id']}:{b['kind']}:{b['idx']}")),
                vector=v,
                payload={"paper_id": b["paper_id"], "year": b.get("year"), "kind": b["kind"], "text": b["text"]},
            ) for b, v in zip(batch, vecs)]
            _qd.upsert(COLL, points=pts)


def count_chunks(pid: str, kind: str | None = None) -> int:
    must = [FieldCondition(key="paper_id", match=MatchValue(value=pid))]
    if kind:
        must.append(FieldCondition(key="kind", match=MatchValue(value=kind)))
    return _qd.count(COLL, count_filter=Filter(must=must), exact=True).count


def search_chunks(vec: list[float], k: int = 20, year_from: int | None = None,
                  paper_ids: list[str] | None = None) -> list[dict]:
    must = []
    if year_from:
        must.append(FieldCondition(key="year", range=Range(gte=year_from)))
    if paper_ids:
        must.append(FieldCondition(key="paper_id", match=MatchAny(any=paper_ids)))
    with tracing.span("vector search", as_type="retriever",
                      input={"k": k, "year_from": year_from, "paper_ids": paper_ids}) as obs:
        res = _qd.query_points(COLL, query=vec, query_filter=Filter(must=must) if must else None,
                               limit=k, with_payload=True)
        hits = [{"score": p.score, **p.payload} for p in res.points]
        obs.update(output=[{"score": round(h["score"], 3), "paper_id": h["paper_id"], "kind": h["kind"],
                            "text": h["text"][:200]} for h in hits])
    return hits
