import asyncio
import logging
import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from hr_assistant.application.ports import EmbeddingProvider
from hr_assistant.core.config import Settings

logger = logging.getLogger(__name__)


class QdrantResumeStore:
    DENSE_VECTOR_NAME = "dense"
    BM25_VECTOR_NAME = "bm25"

    def __init__(self, settings: Settings, embeddings: EmbeddingProvider) -> None:
        self.settings = settings
        self.embeddings = embeddings
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else ""
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key or None,
        )
        self._sparse_encoder: SparseTextEmbedding | None = None

    def _get_sparse_encoder(self) -> SparseTextEmbedding:
        if self._sparse_encoder is None:
            self._sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_encoder

    async def ensure_collection(self) -> None:
        if not await self.client.collection_exists(self.settings.qdrant_collection):
            await self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config={
                    self.DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.settings.openai_embedding_dimensions,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.BM25_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Index the filter fields used by deletion and re-indexing.

        The calls are idempotent from the caller's point of view: Qdrant may
        reject an index that already exists, which is not an error here.
        """

        for field_name in ("candidate_id", "resume_id"):
            try:
                await self.client.create_payload_index(
                    collection_name=self.settings.qdrant_collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Qdrant payload index %s already exists", field_name)

    @staticmethod
    def _filter(field: str, value: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))]
        )

    async def delete_resume(self, resume_id: str) -> None:
        await self._delete_by_filter("resume_id", resume_id)

    async def delete_candidate(self, candidate_id: str) -> None:
        await self._delete_by_filter("candidate_id", candidate_id)

    async def _delete_by_filter(self, field: str, value: str) -> None:
        if not await self.client.collection_exists(self.settings.qdrant_collection):
            return
        await self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.FilterSelector(filter=self._filter(field, value)),
            wait=True,
        )

    async def reassign_candidate(self, source_id: str, target_id: str) -> None:
        if not await self.client.collection_exists(self.settings.qdrant_collection):
            return
        await self.client.set_payload(
            collection_name=self.settings.qdrant_collection,
            payload={"candidate_id": target_id},
            points=self._filter("candidate_id", source_id),
            wait=True,
        )

    async def index_resume(
        self,
        *,
        resume_id: str,
        candidate_id: str,
        source_filename: str,
        chunks: list[str],
    ) -> None:
        if not chunks:
            return
        dense_vectors = await self.embeddings.embed_texts(chunks)
        sparse_vectors = await asyncio.to_thread(
            lambda: list(self._get_sparse_encoder().passage_embed(chunks))
        )
        points: list[models.PointStruct] = []
        for index, (chunk, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hr-resume:{resume_id}:{index}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_VECTOR_NAME: dense,
                        self.BM25_VECTOR_NAME: models.SparseVector(
                            indices=list(sparse.indices), values=list(sparse.values)
                        ),
                    },
                    payload={
                        "candidate_id": candidate_id,
                        "resume_id": resume_id,
                        "source_filename": source_filename,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )
        await self.ensure_collection()
        await self.client.upsert(
            collection_name=self.settings.qdrant_collection, points=points, wait=True
        )

    async def search(
        self, query: str, limit: int, candidate_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        dense_vectors = await self.embeddings.embed_texts([query])
        sparse_vectors = await asyncio.to_thread(
            lambda: list(self._get_sparse_encoder().query_embed(query))
        )
        if not dense_vectors or not sparse_vectors:
            return []
        if not await self.client.collection_exists(self.settings.qdrant_collection):
            return []

        query_filter = None
        if candidate_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="candidate_id", match=models.MatchAny(any=candidate_ids)
                    )
                ]
            )
        prefetch_limit = max(limit, self.settings.qdrant_prefetch_limit)
        result = await self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vectors[0],
                    using=self.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_vectors[0].indices),
                        values=list(sparse_vectors[0].values),
                    ),
                    using=self.BM25_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append({**payload, "score": point.score})
        return hits

    async def close(self) -> None:
        await self.client.close()
