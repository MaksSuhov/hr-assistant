from hr_assistant.container import ServiceContainer
from hr_assistant.infrastructure.db.models import CandidateRecord, ResumeDocumentRecord
from tests.support import json_response


async def _add_document(
    container: ServiceContainer,
    *,
    resume_id: str,
    candidate_id: str,
    text: str,
    filename: str = "resume.pdf",
) -> None:
    async with container.database.unit_of_work() as uow:
        candidate = await uow.candidates.get(candidate_id)
        if candidate is None:
            await uow.candidates.add(
                CandidateRecord(
                    id=candidate_id, full_name=f"Candidate {candidate_id}", status="new"
                )
            )
        await uow.resumes.add(
            ResumeDocumentRecord(
                id=resume_id,
                candidate_id=candidate_id,
                original_filename=filename,
                storage_path=f"/tmp/{resume_id}.pdf",
                uploaded_by="hr@example.org",
                extracted_text=text,
                indexing_status="indexed",
            )
        )
        await uow.commit()


async def test_vector_search_failure_falls_back_to_lexical_matches(
    container: ServiceContainer, fake_vector_store
):
    await _add_document(
        container, resume_id="r1", candidate_id="c1", text="Python FastAPI developer"
    )
    await _add_document(container, resume_id="r2", candidate_id="c2", text="Python developer")
    fake_vector_store.fail_search = True

    hits = await container.search.search_raw("FastAPI", limit=5)

    assert [hit["resume_id"] for hit in hits] == ["r1"]
    assert hits[0]["score"] == 1.0


async def test_lexical_fallback_ignores_too_short_terms(container, fake_vector_store):
    fake_vector_store.fail_search = True

    assert await container.search.search_raw("a b", limit=5) == []


async def test_rerank_reorders_hits_and_normalizes_scores(container, fake_vector_store, fake_llm):
    container.settings.reranking_enabled = True
    fake_llm.configured = True
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["Python developer", "FastAPI developer"],
    )
    fake_llm.completions.responses.append(
        json_response({"ranking": [{"index": 1, "score": 90}, {"index": 0, "score": 10}]})
    )

    hits = await container.search.search_raw("Python FastAPI", limit=2)

    assert [hit["text"] for hit in hits] == ["FastAPI developer", "Python developer"]
    assert hits[0]["score"] == 0.9
    assert fake_llm.completions.calls[0]["response_format"] == {"type": "json_object"}


async def test_rerank_keeps_hybrid_order_when_llm_fails(container, fake_vector_store, fake_llm):
    container.settings.reranking_enabled = True
    fake_llm.configured = True
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["Python developer", "FastAPI developer"],
    )

    hits = await container.search.search_raw("Python FastAPI", limit=2)

    assert {hit["text"] for hit in hits} == {"Python developer", "FastAPI developer"}


async def test_hyde_appends_hypothetical_document_to_the_search_query(
    container: ServiceContainer, fake_vector_store, fake_llm
):
    container.settings.hyde_enabled = True
    fake_llm.configured = True
    fake_llm.completions.responses.append(json_response("hypothetical FastAPI resume"))

    await container.search.search_raw("backend engineer", limit=1)

    assert fake_vector_store.queries[0][0] == "backend engineer\nhypothetical FastAPI resume"


async def test_hyde_failure_keeps_original_query(container, fake_vector_store, fake_llm):
    container.settings.hyde_enabled = True
    fake_llm.configured = True
    await fake_vector_store.index_resume(
        resume_id="r1",
        candidate_id="c1",
        source_filename="resume.pdf",
        chunks=["backend engineer"],
    )

    hits = await container.search.search_raw("backend engineer", limit=1)

    assert hits
    assert fake_vector_store.queries[0][0] == "backend engineer"


async def test_search_maps_candidates_and_deduplicates_chunks(
    container: ServiceContainer, fake_vector_store
):
    await _add_document(container, resume_id="r1", candidate_id="c1", text="Python")
    fake_vector_store.points.extend(
        [
            {
                "candidate_id": "c1",
                "resume_id": "r1",
                "source_filename": "resume.pdf",
                "chunk_index": 0,
                "text": "Python developer",
            },
            {
                "candidate_id": "c1",
                "resume_id": "r1",
                "source_filename": "resume.pdf",
                "chunk_index": 1,
                "text": "Python and FastAPI",
            },
        ]
    )

    hits = await container.search.search("Python", limit=5)

    assert len(hits) == 1
    assert hits[0].candidate_id == "c1"
    assert hits[0].candidate_name == "Candidate c1"
    assert hits[0].document_id == "r1"
