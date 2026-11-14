from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    source: str
    page: int
    document_hash: str
    chunk_index: int


@dataclass(frozen=True)
class Passage:
    text: str
    source: str
    page: int
    distance: float
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    text: str
    passages: list[Passage]
