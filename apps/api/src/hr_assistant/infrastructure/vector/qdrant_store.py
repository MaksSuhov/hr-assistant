import asyncio
import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from hr_assistant.application.ports import EmbeddingProvider
from hr_assistant.core.config import Settings


class QdrantResumeStore:
    DENSE_VECTOR_NAME = "dense"
    BM25_VECTOR_NAME = "bm25"

    def __init__(self, settings: Settings, embeddings: EmbeddingProvider) -> None:
        self.settings = settings
        self.embeddings = embeddings
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else ""
        )
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
        if await self.client.collection_exists(self.settings.qdrant_collection):
            return
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
        await self.client.create_payload_index(
            collection_name=self.settings.qdrant_collection,
            field_name="candidate_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
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
