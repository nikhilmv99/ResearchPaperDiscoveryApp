"""Read recorded traces back from Langfuse for the admin dashboard."""
import json
from datetime import datetime, timedelta, timezone
from statistics import median

from . import config, tracing

QUERY_NAMES = {"search", "ask", "compare","blocked"}  # the trace names used in main.py
MAX_STEPS = 3000
PAGE = 1000
_OPTS = {"timeout_in_seconds": 30}


def _parse(v):
    """Langfuse returns input/output as raw strings."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _seconds(o) -> float | None:
    return (o.end_time - o.start_time).total_seconds() if o.end_time else None


def _level(o) -> str:
    return str(getattr(o.level, "value", o.level) or "DEFAULT")


def _tokens(o) -> tuple[int, int]:
    u = o.usage_details or {}
    return int(u.get("input", 0)), int(u.get("output", 0))


def _fetch(max_items: int, **params) -> tuple[list, bool]:
    """Returns (observations, truncated)."""
    rows, more = [], False
    pages = tracing.client().iter_observation_pages(limit=min(PAGE, max_items), max_items=max_items,
                                                    request_options=_OPTS, **params)
    for page in pages:
        rows.extend(page.data)
        more = bool(page.meta.cursor)
    return rows, more


def _p95(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[max(0, -(-95 * len(vals) // 100) - 1)]  # nearest rank


def list_traces(hours: int, limit: int) -> dict:
    if tracing.client() is None:
        return {
            "enabled": False,
            "traces": [],
            "truncated": False,
            "steps": [],
        }
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    roots, more_roots = _fetch(limit, is_root_observation=True, from_start_time=since,to_start_time=now,
                               fields="core,basic,io")
    steps, more_steps = _fetch(MAX_STEPS, from_start_time=since,to_start_time=now, fields="core,basic,usage",)
    roots = [r for r in roots if r.name in QUERY_NAMES]
    ids = {r.trace_id for r in roots}
    root_ids = {r.id for r in roots}

    by_trace: dict[str, list] = {i: [] for i in ids}
    durations: dict[tuple[str, str], list[float]] = {}
    for s in steps:
        if s.trace_id not in ids:
            continue
        by_trace[s.trace_id].append(s)
        d = _seconds(s)
        if s.id not in root_ids and d is not None:
            durations.setdefault((s.name or "?", s.type), []).append(d)

    out = []
    for r in sorted(roots, key=lambda r: r.start_time, reverse=True):
        kids = [s for s in by_trace[r.trace_id] if s.id != r.id]
        tok = [_tokens(s) for s in kids]
        inp = _parse(r.input)
        out.append({
            "trace_id": r.trace_id, "name": r.name, "user": r.user_id, "time": r.start_time.isoformat(),
            "question": (
    inp.get("question")
    or f"Blocked: {inp.get('endpoint', '')} - Rule: {inp.get('rule', '')}"
    if isinstance(inp, dict)
    else str(inp or "")),
            "seconds": _seconds(r), "failed": r.name == "blocked" or _level(r) == "ERROR", "error": r.status_message,
            "step_problems": sum(_level(s) in ("ERROR", "WARNING") for s in kids),
            "llm_calls": sum(s.type == "GENERATION" for s in kids),
            "tokens_in": sum(t[0] for t in tok), "tokens_out": sum(t[1] for t in tok),
        })
    return {
        "traces": out, "truncated": more_roots or more_steps,
        "steps": [{"name": n, "type": t, "count": len(v), "median_s": median(v), "p95_s": _p95(v)}
                  for (n, t), v in durations.items()],
    }


def get_trace(trace_id: str) -> dict | None:
    if tracing.client() is None:
        return None
    rows, _ = _fetch(PAGE, trace_id=trace_id, fields="core,basic,io,metadata,model,usage")
    if not rows:
        return None
    known = {o.id for o in rows}
    children: dict[str | None, list] = {}
    for o in rows:
        parent = o.parent_observation_id if o.parent_observation_id in known else None
        children.setdefault(parent, []).append(o)
    t0 = min(o.start_time for o in rows)
    steps = []

    def walk(parent: str | None, depth: int) -> None:
        for o in sorted(children.get(parent, []), key=lambda o: o.start_time):
            tin, tout = _tokens(o)
            steps.append({
                "id": o.id, "name": o.name or "?", "type": o.type, "depth": depth,
                "start_s": (o.start_time - t0).total_seconds(), "seconds": _seconds(o),
                "level": _level(o), "status": o.status_message, "model": o.model,
                "tokens_in": tin, "tokens_out": tout,
                "input": _parse(o.input), "output": _parse(o.output),
                "metadata": {k: _parse(v) for k, v in (o.metadata or {}).items()},
            })
            walk(o.id, depth + 1)

    walk(None, 0)
    try:
        url = tracing.client().get_trace_url(trace_id=trace_id)
    except Exception:
        url = None
    return {"trace_id": trace_id, "steps": steps, "url": url, "host": config.LANGFUSE_BASE_URL}