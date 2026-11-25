"""Guard rails for the research agent (search, ask and compare).

  before the run  check_input(), slot()                  bad questions, secrets, per-user limits
  planning        clean_plan()                           scope check, validates what the LLM planned
  retrieved text  mask_injections(), fence()             papers are data, never instructions
  every LLM call  neutralize()                           no chat-template tokens from users or papers
  after the run   check_answer(), check_comparison()     citations, leaks, remote images, made-up papers

Downloads of paper PDFs are guarded in safe_fetch.py.

The text patterns are a first filter and can be worded around. What contains an attack that gets past
them is the rest: fenced evidence, validated output, no remote images, and tools with fixed targets.
"""
import logging
import re
import threading
import time
import unicodedata
from collections import deque
from contextlib import contextmanager
from datetime import date

from . import config, tracing

log = logging.getLogger(__name__)


class Blocked(Exception):
    """A guard rail stopped the request. The message is shown to the user."""

    def __init__(self, rule: str, message: str, status: int = 400):
        super().__init__(message)
        self.rule, self.message, self.status = rule, message, status


# ---------- text cleaning ----------

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Zero-width, bidi-override and Unicode "tag" characters: invisible to people, readable by the model.
_INVISIBLE = re.compile("[­​‎‏‪-‮⁠-⁤⁦-⁩﻿"
                        "\U000e0000-\U000e007f]")


def clean_text(text) -> str:
    return _INVISIBLE.sub("", _CTRL.sub(" ", text)).strip() if isinstance(text, str) else ""


def _match_form(text: str) -> str:
    """Only used for matching: folds look-alike letters and drops every invisible character."""
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Cf").replace("’", "'")


def neutralize(text: str) -> str:
    """Break chat-template tokens such as <|im_start|> so text cannot open a fake system turn."""
    return text.replace("<|", "< |").replace("|>", "| >")


_IMAGE = re.compile(r"!+\[")
_LINK = re.compile(r"\[([^\[\]]*)\]\([^()]*\)")


def clean_display(text, limit: int | None = None, links: bool = True) -> str:
    """Text the UI renders as Markdown. Images are disabled because the browser loads them without a
    click, so an injected ![](https://evil/?q=...) could send data out."""
    text = clean_text(text)
    while True:  # removing one link can form another one
        new = _IMAGE.sub("[", text)
        if not links:
            new = _LINK.sub(r"\1", new)
        if new == text:
            break
        text = new
    return text if limit is None else text[:limit]


# ---------- before the run: the question ----------

_SECRET = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b(?:sk|pk|rk)-(?:[A-Za-z]+-)*[A-Za-z0-9_]{20,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"
    r"|\bAKIA[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{35}|\bxox[abprs]-[A-Za-z0-9-]{10,}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

_OVERRIDE = (r"(?:ignore|disregard|forget|override|bypass)\s+(?:\w+\s+){0,3}?(?:previous|prior|above|earlier|"
             r"preceding|your|system)\s+(?:\w+\s+){0,2}?(?:instructions?|prompts?|rules|guidelines|guardrails)")
_REVEAL = (r"(?:reveal|print|show|repeat|output|display|tell me|give me)\s+(?:\w+\s+){0,3}?"
           r"(?:(?:system|hidden|initial|developer)\s+(?:prompt|instructions?|message)"
           r"(?=\s*(?:$|[.!?,;:\n]|and\b|above\b|verbatim\b|word\b|to me\b))|your\s+(?:prompt|instructions?)\b)")
_PERSONA = r"(?:you are now|act as|switch to|enter)\s+(?:\w+\s+){0,2}?(?:dan|developer mode|jailbroken|unrestricted|unfiltered)\b"

# Questions: only a command at the start of a sentence counts, and quoted text is skipped, so questions
# about prompt injection ("defenses against 'ignore previous instructions' attacks") still work.
_COMMAND = re.compile(r"(?:^|[.!?;:,\n]|\b(?:and|then|now|please|instead))\s*(?:please\s+|now\s+|just\s+)*"
                      rf"(?:{_OVERRIDE}|{_REVEAL}|{_PERSONA})", re.I)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|(?<!\w)'[^']*'(?!\w)")


def check_input(text: str) -> str:
    """The cleaned question, or Blocked. Call it before anything is traced, stored or sent out."""
    text = clean_text(text)
    if not text:
        raise Blocked("empty", "Please enter a question.")
    probe = _match_form(text)
    if _SECRET.search(probe) or tracing.redact(probe) != probe:
        raise Blocked("secret", "Your question seems to contain a password or API key. Remove it and try again.")
    if _COMMAND.search(_QUOTED.sub(" ", probe)):
        raise Blocked("injection", "That looks like an instruction to the assistant, not a research question. "
                                   "If you are asking about such attacks, put the example in quotes.")
    return _EMAIL.sub("[email]", text)


# ---------- before the run: per-user limits ----------

_lock = threading.Lock()
_history: dict[str, deque] = {}  # request times of the last 24 hours (in memory: the app runs ONE worker)
_active: dict[str, int] = {}
DAY = 86400


@contextmanager
def slot(user: str):
    """Hold one agent run for this user. The LLM serves one request at a time, so one user must not fill it."""
    now = time.monotonic()
    with _lock:
        times = _history.setdefault(user, deque())
        while times and now - times[0] > DAY:
            times.popleft()
        if _active.get(user, 0) >= config.GUARD_CONCURRENT:
            raise Blocked("concurrency", "Your previous request is still running. Wait for it to finish.", 429)
        if sum(now - t < 60 for t in times) >= config.GUARD_PER_MINUTE:
            raise Blocked("rate", "Too many requests. Wait a minute and try again.", 429)
        if len(times) >= config.GUARD_PER_DAY:
            raise Blocked("daily", "You have reached the daily request limit.", 429)
        times.append(now)
        _active[user] = _active.get(user, 0) + 1
    try:
        yield
    finally:
        with _lock:
            _active[user] -= 1
            if not _active[user]:
                del _active[user]


# ---------- planning ----------

MAX_KEYWORDS = 5


def _clean_keyword(k) -> str:
    """Keywords go to public search APIs and into a quoted arXiv query, so quotes must not survive."""
    k = re.sub(r"[\"\\\s]+", " ", tracing.redact(clean_text(k))).strip()
    if "://" in k or _EMAIL.search(k) or _SECRET.search(k):
        return ""
    return " ".join(k.split()[:8])[:80]


def clean_plan(raw: dict, question: str) -> dict:
    """Check the planner's JSON. Anything missing or malformed falls back to the user's own words."""
    if raw.get("in_scope") is False:  # only an explicit false blocks: a malformed reply must not
        raise Blocked("scope", "I can only search for research papers. Try asking about a research topic.")
    kws = raw.get("keywords")
    kws = [c for k in (kws if isinstance(kws, list) else []) if (c := _clean_keyword(k))][:MAX_KEYWORDS]
    yf = raw.get("year_from")
    ok = isinstance(yf, int) and not isinstance(yf, bool) and 1950 <= yf <= date.today().year
    return {"semantic_query": clean_text(raw.get("semantic_query"))[:500] or question,
            "keywords": kws or [_clean_keyword(question) or question[:80].replace('"', " ")],
            "year_from": yf if ok else None}


# ---------- retrieved text ----------

# Papers: matched anywhere, to the end of the sentence. Includes the hidden notes to AI reviewers that
# have turned up in real preprints.
_EMBEDDED = re.compile(
    rf"(?:\b(?:{_OVERRIDE}|{_REVEAL}|{_PERSONA}"
    r"|give (?:an? (?:positive|favou?rable) review|(?:positive|favou?rable) reviews? only)"
    r"|do not (?:highlight|mention|report) any (?:negatives|weaknesses|limitations)"
    r"|(?:as an? (?:ai|language model|llm)[, ]+)you (?:must|should|will)"
    r"|do not (?:tell|inform|alert) the user)\b)[^.!?\n]*[.!?]?", re.I)
MASK = "[instruction-like text removed]"


def mask_injections(text: str) -> tuple[str, bool]:
    """(text that is safe to show the model, whether anything was masked)"""
    masked, n = _EMBEDDED.subn(MASK, clean_text(text))
    return masked, bool(n)


def fence(tag: str, nonce: str, label: str, text: str) -> str:
    """Wrap untrusted text. The nonce is random per request, so a paper cannot write its own end marker."""
    label = re.sub(r"[\s<>]+", " ", label).strip()[:300]
    return f"<{tag}-{nonce} {label}>\n{text}\n</{tag}-{nonce}>"


def safe_url(url) -> str:
    url = clean_text(url)
    return url if re.match(r"https?://[^\s()<>\"]+$", url, re.I) and len(url) <= 2000 else ""


def clean_paper(p: dict) -> dict:
    """Metadata from public APIs (and older rows in the database) is untrusted too."""
    return {**p, "title": clean_display(p.get("title"), 300), "authors": clean_display(p.get("authors"), 1000),
            "abstract": clean_display(p.get("abstract"), 5000),
            "url": safe_url(p.get("url")), "pdf_url": safe_url(p.get("pdf_url"))}


# ---------- after the run ----------

_CITE = re.compile(r"\s?\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
_NOT_FOUND = re.compile(r"(?:could not|couldn't|cannot|can't|unable to) (?:find|answer|locate)|not (?:found|stated|mentioned|"
                        r"provided|available)|(?:do|does)(?: not|n't) (?:contain|provide|mention|include|state|discuss)", re.I)
REFUSAL = "I can't answer that. I can only answer questions about the uploaded papers."
EMPTY = "The model returned no answer. Please try again."


def _leaks(text: str, system: str) -> bool:
    flat = " ".join(text.lower().split())
    lines = (" ".join(line.lower().split()).lstrip("- ") for line in system.splitlines())
    return any(len(line) >= 40 and line in flat for line in lines)


def check_answer(answer: str, n_sources: int, system: str) -> tuple[str, list[str]]:
    """(safe answer, warnings for the user)"""
    if _leaks(answer, system):
        log.warning("guard rail: the answer repeated the system prompt")
        return REFUSAL, []
    text = tracing.redact(clean_display(answer, links=False))
    text = _SECRET.sub("[redacted]", text)
    if not text:
        return EMPTY, []

    warnings, bad, cited = [], [], False

    def cite(m):
        nonlocal cited
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if 0 in nums:  # an interval like [0, 1], not a citation
            return m.group(0)
        if any(n <= n_sources for n in nums):
            cited = True
            return m.group(0)
        bad.append(m.group(1))
        return ""

    text = _CITE.sub(cite, text).strip()
    if bad:
        warnings.append("Removed citations that do not match any source: " + ", ".join(f"[{b}]" for b in bad[:5]))
    if not cited and not _NOT_FOUND.search(text):
        warnings.append("This answer cites no source. Check it against the papers.")
    return text or EMPTY, warnings


FIELDS = ("objective", "method", "data", "evaluation", "limitations")


def check_comparison(raw, papers: list[dict]) -> dict:
    """Keep only the papers the user selected, under their real titles, and only the expected fields."""
    rows = raw.get("papers") if isinstance(raw, dict) else None
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def cell(v):
        v = ", ".join(map(str, v)) if isinstance(v, list) else "" if v is None else str(v)
        return tracing.redact(clean_display(v, 400, links=False)) or "Not stated"

    def find(p):
        title = clean_text(p["title"]).lower()
        return (next((r for r in rows if r.get("id") == p["paper_id"]), None)
                or next((r for r in rows if title and clean_text(r.get("title")).lower() == title), None))

    found = [(p, find(p)) for p in papers]
    if not any(r for _, r in found):
        return {"error": "The model did not return the selected papers"}
    out = {"papers": [{"id": p["paper_id"], "title": clean_display(p["title"], 300),
                       **{f: cell((r or {}).get(f)) for f in FIELDS}} for p, r in found],
           "summary": tracing.redact(clean_display(str(raw.get("summary") or ""), 1000, links=False))}
    missing = [clean_display(p["title"], 80) for p, r in found if not r]
    if missing:
        out["warnings"] = ["The model returned nothing for: " + ", ".join(missing)]
    return out
