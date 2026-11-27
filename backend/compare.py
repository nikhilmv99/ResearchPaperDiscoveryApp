"""Paper Analysis/Compare Agent: retrieve evidence per paper and aspect, then one LLM call.

Budget: the endpoint allows 8192 tokens input+output, so evidence is capped at ~12k chars
(~3k tokens) and the answer at 1024 tokens.
"""
from . import embeddings, ingest, llm, store
import secrets
import json
import re

from . import guardrails, tracing
ASPECTS = {
    "objective": "research objective and problem addressed",
    "method": "methodology, approach, model architecture",
    "data": "dataset, data used, embeddings or features",
    "evaluation": "evaluation metrics, experiments and results",
    "limitations": "limitations, weaknesses and future work",
}
MAX_PAPERS = 3
EVIDENCE_CHARS = 12000

SYSTEM = """You compare research papers using ONLY the evidence provided. If the evidence does not
say something, write "Not stated". Reply with JSON only:
{"papers":[{"id":"...","title":"...","objective":"","method":"","data":"","evaluation":"","limitations":""}],
 "summary":"2-3 sentences on the key differences"}
Keep every field under 30 words."""


def compare_papers(paper_ids: list[str], focus: str | None = None) -> dict:
    papers = store.get_papers(paper_ids[:MAX_PAPERS])
    if len(papers) < 2:
        raise ValueError("Select at least two papers from the library")

    with tracing.span("fetch full text", as_type="tool",
                      input={"paper_ids": [p["paper_id"] for p in papers]}) as obs:
        for p in papers:
            ingest.ensure_fulltext(p)
        obs.update(output={p["paper_id"]: store.count_chunks(p["paper_id"], "fulltext") for p in papers})

    vecs = embeddings.embed([f"{q}. {focus}" if focus else q for q in ASPECTS.values()])
    per_cell = EVIDENCE_CHARS // (len(papers) * len(ASPECTS))
    nonce = secrets.token_hex(8)
    flagged = []
    blocks = []
    retrieved_count = 0

    with tracing.span(
        "comparison evidence retrieval",
        as_type="retriever",
        input={
            "paper_ids": [p["paper_id"] for p in papers],
            "aspects": list(ASPECTS.keys()),
        },
    ) as obs:

        for p in papers:
            title, title_hit = guardrails.mask_injections(p["title"])
            paper_flagged = title_hit
            lines = []

            for (aspect, _), vec in zip(ASPECTS.items(), vecs):
                hits = store.search_chunks(
                    vec,
                    k=2,
                    paper_ids=[p["paper_id"]],
                )

                retrieved_count += len(hits)

                text, masked = guardrails.mask_injections(
                    " ".join(h["text"] for h in hits)
                )

                paper_flagged = paper_flagged or masked

                lines.append(
                    f"[{aspect}] {text[:per_cell]}"
                )

            blocks.append(
                guardrails.fence(
                    "paper",
                    nonce,
                    f"id={p['paper_id']} title={title}",
                    "\n".join(lines),
                )
            )

            if paper_flagged:
                flagged.append(
                    guardrails.clean_display(p["title"], 80)
                )

        obs.update(output={
            "paper_count": len(papers),
            "retrieved_chunks": retrieved_count,
            "masked_papers": len(flagged),
        })

    user = (f"Focus: {focus}\n\n" if focus else "") + "\n\n".join(blocks)
    text = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    max_tokens=1024, temperature=0.1)
    safe_raw = tracing.redact(guardrails.clean_display(text, 500, links=False))
    import json, re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"error": "No JSON returned", "raw": safe_raw}

    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {
            "error": "Could not parse model output",
            "raw": safe_raw,
        }

    res = guardrails.check_comparison(raw, papers)

    if flagged:
        res.setdefault("warnings", []).append(
            "Instruction-like text was hidden from: "
            + ", ".join(flagged)
        )

    return res