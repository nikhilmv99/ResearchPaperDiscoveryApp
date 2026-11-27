"""LangGraph workflow:  plan (LLM) -> local vector search -> [optional web search + index -> local vector search]."""
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from functools import wraps
from . import guardrails, tracing, config
from . import embeddings, llm, sources, store

MIN_SCORE = 0.7


class State(TypedDict, total=False):
    question: str
    use_web: bool
    plan: dict
    fresh_ids: list[str]
    results: list[dict]


PLAN_PROMPT = f"""You plan searches for a research-paper search engine. Today's year is {date.today().year}.
Reply with JSON only, no other text, with these keys:
"semantic_query": one clear sentence describing the ideal paper (used for embedding search),
"keywords": 3 to 5 short search phrases (1-4 words) using the technical terms, synonyms and acronyms papers use,
"year_from": an integer year only if the user implies recency or a date range, otherwise null.
"in_scope": true only if the user wants academic research papers on a topic. false for requests for instructions 
to make weapons, explosives or other harmful items, for chit-chat, or anything that is not a paper search. Studying 
a harmful topic academically (detection, safety, history) is in scope."""

def _brief(out: State) -> dict:
    if "results" in out:
        return {
            "results": [
                {
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "score": p["score"],
                }
                for p in out["results"]
            ]
        }

    return {
        k: out[k]
        for k in ("plan", "fresh_ids")
        if k in out
    }


def _traced(name: str):
    def deco(fn):
        @wraps(fn)
        def node(s: State) -> State:
            with tracing.span(
                name,
                as_type="agent",
                input={
                    k: v
                    for k, v in s.items()
                    if k in ("question", "plan", "fresh_ids")
                },
            ) as obs:

                out = fn(s)
                obs.update(output=_brief(out))
                return out

        return node
    return deco

@_traced("plan")
def plan_node(s: State) -> State:
    q = s["question"]
    raw = llm.chat_json(
        [{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": q}],
        max_tokens=300,
        temperature=0.1,
    )
    return {"plan": guardrails.clean_plan(raw, q)}


def _run_local_search(plan: dict, fresh_ids: list[str], min_score: float = MIN_SCORE) -> list[dict]:
    vec = embeddings.embed([plan["semantic_query"]])[0]
    hits = store.search_chunks(vec, k=40, year_from=plan["year_from"])

    best: dict[str, dict] = {}
    for h in hits:
        score = h.get("score")
        if score is None or score < min_score:
            continue
        # hits are sorted by score descending, so first qualifying hit per paper is best
        best.setdefault(h["paper_id"], h)

    if not best:
        return []

    meta = {
        p["paper_id"]: guardrails.clean_paper(p)
        for p in store.get_papers(list(best))
    }
    fresh = set(fresh_ids)
    results = [
        {
            **meta[pid],
            "score": round(h["score"], 3),
            "snippet": guardrails.clean_display(h["text"][:300]),
            "fresh": pid in fresh,
        }
        for pid, h in best.items()
        if pid in meta
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:10]

@_traced("local search")
def local_node(s: State) -> State:
    results = _run_local_search(s["plan"], s.get("fresh_ids", []), min_score=MIN_SCORE)
    return {"results": results}

@_traced("web search")
def web_node(s: State) -> State:
    plan = s["plan"]
    papers = sources.web_search(
        plan["keywords"],
        plan["year_from"],
        time_budget=config.GUARD_WEB_SECONDS,
    )

    papers = [
        guardrails.clean_paper(p)
        for p in papers
    ]
    new = [p for p in papers if not store.has_paper(p["paper_id"])]
    for p in new:
        store.upsert_paper(p)
    if new:
        store.add_chunks([
            {
                "paper_id": p["paper_id"],
                "year": p["year"],
                "kind": "abstract",
                "idx": 0,
                "text": f"{p['title']}. {p['abstract']}",
            }
            for p in new
        ])
    return {"fresh_ids": [p["paper_id"] for p in new]}


def after_local(s: State) -> str:
    return "web" if s.get("use_web", True) else END


_g = StateGraph(State)
_g.add_node("plan", plan_node)
_g.add_node("local", local_node)
_g.add_node("web", web_node)
_g.add_node("local_after_web", local_node)

_g.add_edge(START, "plan")
_g.add_edge("plan", "local")
_g.add_conditional_edges("local", after_local, {"web": "web", END: END})
_g.add_edge("web", "local_after_web")
_g.add_edge("local_after_web", END)

search_graph = _g.compile()


def run_search(question: str, use_web: bool = True) -> dict:
    out = search_graph.invoke({"question": question, "use_web": use_web})
    return {"plan": out["plan"], "results": out.get("results", [])}