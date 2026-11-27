"""PDF -> text -> chunks -> vector store."""
import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from . import safe_fetch
from . import store

log = logging.getLogger(__name__)


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + size] for i in range(0, len(text), size - overlap)] if text else []


def pdf_to_text(data: bytes) -> str:
    import pymupdf  # PyMuPDF
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def ingest_fulltext(paper: dict, text: str, max_chunks: int = 240) -> int:
    chunks = chunk_text(text)[:max_chunks]
    store.add_chunks([{"paper_id": paper["paper_id"], "year": paper["year"],
                       "kind": "fulltext", "idx": i, "text": c} for i, c in enumerate(chunks)])
    return len(chunks)


def ingest_upload(filename: str, data: bytes) -> dict:
    text = pdf_to_text(data)
    if not text.strip():
        raise ValueError("No text found in PDF (scanned PDFs need OCR)")

    yr: int | None = None
    ref_split = re.split(r"\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\b", text, maxsplit=1)
    body_text = ref_split[0]
    header_text = body_text[:3000]

    # 1. Check for modern arXiv ID in main body (e.g., arXiv:2611.14584v1)
    arxiv_match = re.search(r"arXiv\s*:\s*(\d{2})\d{2}\.\d+", body_text, re.IGNORECASE)
    if arxiv_match:
        yy = int(arxiv_match.group(1))
        yr = 2000 + yy if yy <= 90 else 1900 + yy

    # 2. Check for explicit publication / copyright / conference lines near top
    if yr is None:
        patterns = [
            r"(?:Published|Accepted|Available online)[^0-9\n]{0,25}((?:19|20)\d{2})",
            r"(?:©|\(c\)|Copyright)\s*((?:19|20)\d{2})",
            r"(?:Proceedings|Conference|Symposium|Journal|Transactions|IEEE|ACM|Springer|Elsevier)[^\n]{0,40}((?:19|20)\d{2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                yr = int(match.group(1))
                break

    # 3. Check explicit date in body (e.g., "10 Nov 2026")
    if yr is None:
        date_match = re.search(
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+((?:19|20)\d{2})\b",
            body_text[:5000],
            re.IGNORECASE,
        )
        if date_match:
            yr = int(date_match.group(1))

    # 4. Last-resort fallback only if still None
    if yr is None:
        max_yr = datetime.now(timezone.utc).year + 1
        footer_text = text[-6000:]
        ref_years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", footer_text)]
        valid_years = [y for y in ref_years if 1950 <= y <= max_yr]
        if valid_years:
            yr = max(valid_years)

    paper = {
        "paper_id": "upload:" + hashlib.sha1(data).hexdigest()[:12],
        "title": Path(filename).stem,
        "authors": "",
        "year": yr,
        "abstract": re.sub(r"\s+", " ", text)[:1500],
        "url": "",
        "pdf_url": "",
        "source": "upload",
        "citations": None,
    }
    store.upsert_paper(paper)
    ingest_fulltext(paper, text)
    return paper

def ensure_fulltext(paper: dict) -> None:
    """Download and index the PDF of a paper if only its abstract is indexed."""
    if store.count_chunks(paper["paper_id"], "fulltext") or not paper.get("pdf_url"):
        return
    try:
        data = safe_fetch.fetch_pdf(paper["pdf_url"])
        ingest_fulltext(paper, pdf_to_text(data))
    except Exception as e:
        log.warning(
            "Full text unavailable for %s: %s",
            paper["paper_id"],
            e,
        )
