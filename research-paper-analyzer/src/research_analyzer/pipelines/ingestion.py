from pathlib import Path

from research_analyzer.services.pdf import PDFTextExtractor
from research_analyzer.services.chunking import PDFChunker


class PDFIngestionPipeline:
    def __init__(
        self,
        embedder,
        store,
        extractor=None,
        chunker=None,
        batch_size: int = 16,
    ):
        if not 1 <= batch_size <= 16:
            raise ValueError("Batch size must be 1..16")
        self.extractor = extractor or PDFTextExtractor()
        self.chunker = chunker or PDFChunker()
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    def ingest(self, folder: str) -> dict:
        paths = sorted(Path(folder).rglob("*.pdf"))
        report = {"files": len(paths), "added_chunks": 0, "errors": []}
        for path in paths:
            try:
                pages = self.extractor.extract(path)
                if not pages:
                    report["errors"].append(
                        f"{path}: no selectable text (OCR required)"
                    )
                    continue
                chunks = self.chunker.split(path, pages)
                for start in range(0, len(chunks), self.batch_size):
                    batch = chunks[start:start + self.batch_size]
                    existing = self.store.existing_ids(
                        [chunk.id for chunk in batch]
                    )
                    batch = [chunk for chunk in batch if chunk.id not in existing]
                    if batch:
                        vectors = self.embedder.embed(
                            [chunk.text for chunk in batch]
                        )
                        self.store.save(batch, vectors)
                        report["added_chunks"] += len(batch)
            except Exception as exc:
                report["errors"].append(f"{path}: {exc}")
        return report
