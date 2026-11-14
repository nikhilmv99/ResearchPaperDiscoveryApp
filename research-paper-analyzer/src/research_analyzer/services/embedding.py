from pathlib import Path
import httpx
from openai import OpenAI
from research_analyzer.config import Settings


class EmbeddingService:
    """OpenAI-compatible hosted BGE endpoint, batches of at most 16."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.model = settings.embedding_model
        if client is not None:
            self.client = client
            return
        if not settings.embedding_api_key or not settings.embedding_ca_cert:
            raise ValueError("Set EMBEDDING_API_KEY and EMBEDDING_CA_CERT")
        if not Path(settings.embedding_ca_cert).is_file():
            raise FileNotFoundError(settings.embedding_ca_cert)
        http_client = httpx.Client(verify=settings.embedding_ca_cert, timeout=60)
        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            http_client=http_client,
            max_retries=3,
        )

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > 16:
            raise ValueError("Maximum 16 texts per embedding request")
        if is_query:
            texts = [self.QUERY_PREFIX + text for text in texts]
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]
        if len(vectors) != len(texts) or any(len(vector) != 384 for vector in vectors):
            raise ValueError("Expected one 384-dimensional vector per text")
        return vectors
