from pathlib import Path
import fitz


class PDFTextExtractor:
    """Extract selectable text, preserving PDF page numbers."""

    def extract(self, path: Path) -> list[tuple[int, str]]:
        pages = []
        with fitz.open(path) as doc:
            for index, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append((index + 1, text))
        return pages
