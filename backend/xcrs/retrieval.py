"""LangChain retriever over our own pgvector queries (ADR-0019).

Not LangChain's PGVector store: that keeps its own tables and only does top-k, while our vectors live in
per-model tables with per-model indexes (ADR-0008/0009) and concept matching must stay a threshold scan
(ADR-0010). This class only gives the existing k-NN course search the standard `retriever.invoke(query)`
interface, so it can serve as a tool for a future chat or agent. The recommendation path doesn't use it:
its candidates come precomputed from `concept_course_matches`.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, SkipValidation
from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel
from xcrs.embeddings.base import Embedder
from xcrs.repository import catalog as catalog_repo
from xcrs.repository import vectors


class CourseRetriever(BaseRetriever):
    """Courses nearest to a free-text query, using the given model's HNSW index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SkipValidation[Session]
    model: SkipValidation[EmbeddingModel]
    embedder: SkipValidation[Embedder]
    k: int = 10

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector = self.embedder.embed_query([query])[0]
        hits = vectors.nearest_courses(self.session, self.model, vector, k=self.k)
        courses = catalog_repo.courses_by_id(self.session, [course_id for course_id, _ in hits])
        return [
            Document(
                page_content=" - ".join(filter(None, [courses[course_id].title, courses[course_id].headline])),
                metadata={
                    "course_id": course_id,
                    "title": courses[course_id].title,
                    "url": courses[course_id].url,
                    "similarity": round(similarity, 4),
                },
            )
            for course_id, similarity in hits
            if courses[course_id].is_active  # switched-off courses are never offered
        ]
