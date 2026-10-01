from xcrs.config import get_settings
from xcrs.db.models import EmbeddingModel
from xcrs.embeddings.base import Embedder
from xcrs.embeddings.openai_compatible import OpenAICompatibleEmbedder


def embedder_for(model: EmbeddingModel, query_prefix: str | None = None) -> Embedder:
    """Build the embedder for a registered model. The endpoint comes from configuration.
    `query_prefix` overrides the model's instruction for a different task (e.g. skill matching)."""
    settings = get_settings()
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        model_id=model.name,
        dimensions=model.dimensions,
        query_prefix=model.query_prefix if query_prefix is None else query_prefix,
        document_prefix=model.document_prefix,
        batch_size=settings.embedding_batch_size,
        timeout_s=settings.embedding_timeout_s,
    )
