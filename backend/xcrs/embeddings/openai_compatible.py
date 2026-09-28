import httpx


class OpenAICompatibleEmbedder:
    """Embedder for any server exposing the OpenAI `/embeddings` API (Ollama, TEI, vLLM, OpenAI)."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        dimensions: int,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        batch_size: int = 32,
        timeout_s: float = 120.0,
        api_key: str | None = None,
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.query_prefix + t for t in texts])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.post("/embeddings", json={"model": self.model_id, "input": batch})
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)

        if len(vectors) != len(texts):
            raise ValueError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"{self.model_id} returned {len(v)} dims, registry says {self.dimensions}")
        return vectors
