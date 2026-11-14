class RAGRetriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int = 5, source: str | None = None):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.store.count() == 0:
            return []
        vector = self.embedder.embed([question], is_query=True)[0]
        return self.store.search(vector, top_k=top_k, source=source)
