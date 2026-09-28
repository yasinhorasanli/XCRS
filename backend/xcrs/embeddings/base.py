from typing import Protocol


class Embedder(Protocol):
    """Model-agnostic embedding interface (ADR-0006).

    Queries (short user phrases) and documents (courses, roadmap concepts) are embedded
    through separate methods, because many models expect them to be prepared differently.
    """

    model_id: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, texts: list[str]) -> list[list[float]]: ...
