import chromadb

from research_analyzer.models import TextChunk, Passage


class ChromaVectorStore:
    def __init__(
        self,
        directory: str = "./chroma_db",
        collection: str = "research_papers",
        client=None,
    ):
        client = client or chromadb.PersistentClient(path=directory)
        self.collection = client.get_or_create_collection(name=collection)

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self, ids: list[str]) -> set[str]:
        return (
            set(self.collection.get(ids=ids, include=[])["ids"])
            if ids
            else set()
        )

    def save(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must match")
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "document_hash": chunk.document_hash,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        source: str | None = None,
    ) -> list[Passage]:
        if self.count() == 0:
            return []
        where = {"source": source} if source else None
        # Chroma accepts n_results greater than a filtered subset and returns
        # the available records.
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Passage(
                text=text,
                source=metadata["source"],
                page=int(metadata["page"]),
                distance=float(distance),
                chunk_id=chunk_id,
            )
            for text, metadata, distance, chunk_id in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                result["ids"][0],
            )
        ]
