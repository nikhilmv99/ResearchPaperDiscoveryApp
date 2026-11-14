import hashlib
from pathlib import Path
from transformers import AutoTokenizer
from research_analyzer.models import TextChunk


class PDFChunker:
    def __init__(self, chunk_size: int = 350, overlap: int = 40, tokenizer=None):
        if not 0 <= overlap < chunk_size <= 480:
            raise ValueError("Require 0 <= overlap < chunk_size <= 480")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, path: Path, pages: list[tuple[int, str]]) -> list[TextChunk]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        chunks = []
        for page, text in pages:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            step = self.chunk_size - self.overlap
            for index, start in enumerate(range(0, len(ids), step)):
                part = self.tokenizer.decode(
                    ids[start:start + self.chunk_size],
                    skip_special_tokens=True,
                ).strip()
                if not part:
                    continue
                chunk_id = hashlib.sha256(
                    f'{path.resolve()}:{digest}:{page}:{index}'.encode()
                ).hexdigest()
                chunks.append(TextChunk(chunk_id, part, path.name, page, digest, index))
        return chunks
