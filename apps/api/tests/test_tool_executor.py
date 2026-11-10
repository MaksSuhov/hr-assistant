import json
from datetime import UTC, datetime

from hr_assistant.application.tool_executor import ToolExecutor
from hr_assistant.schemas.application import ApplicationOut
from hr_assistant.schemas.candidate import CandidateOut
from hr_assistant.schemas.matching import (
    CandidateSearchHit,
    CompareCandidatesResponse,
    MatchAnalysis,
)


class StubSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int = 10) -> list[CandidateSearchHit]:
        self.calls.append((query, limit))
        return [
            CandidateSearchHit(
                candidate_id="c1",
                candidate_name="Candidate",
                document_id="r1",
                source_filename="resume.pdf",
                snippet="Python developer",
                score=0.9,
            )
        ]


class StubCandidates:
    async def get(self, candidate_id: str) -> CandidateOut:
        now = datetime.now(UTC)
        return CandidateOut(
            id=candidate_id,
            full_name="Candidate",
            email=None,
            phone=None,
            created_at=now,
            updated_at=now,
            sources=[],
            applications=[],
        )


class StubApplications:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def list_for_vacancy(self, vacancy_id: str, limit: int = 200) -> list[ApplicationOut]:
        self.calls.append((vacancy_id, limit))
        now = datetime.now(UTC)
        return [
            ApplicationOut(
                id="a1",
                candidate_id="c1",
                candidate_name="Candidate",
                vacancy_id=vacancy_id,
                vacancy_title="Backend",
                stage="interview",
                stage_changed_at=now,
                created_at=now,
                updated_at=now,
            )
        ]


class StubMatching:
    async def compare(self, vacancy_id: str, candidate_ids: list[str]) -> CompareCandidatesResponse:
        return CompareCandidatesResponse(
            vacancy_id=vacancy_id,
            candidates=[
                MatchAnalysis(
                    candidate_id=candidate_id,
                    vacancy_id=vacancy_id,
                    summary="ok",
                    requirement_evidence=[],
                )
                for candidate_id in candidate_ids
            ],
        )


def build_executor() -> tuple[ToolExecutor, StubSearch]:
    search = StubSearch()
    return (
        ToolExecutor(StubCandidates(), search, StubMatching(), StubApplications()),
        search,
    )


async def test_search_tool_clamps_limit():
    executor, search = build_executor()

    payload = json.loads(
        await executor.execute("search_candidates", json.dumps({"query": "python", "limit": 99}))
    )

    assert search.calls == [("python", 10)]
    assert payload[0]["candidate_id"] == "c1"


async def test_search_tool_marks_snippets_as_resume_data():
    executor, _ = build_executor()

    payload = json.loads(
        await executor.execute("search_candidates", json.dumps({"query": "python"}))
    )

    assert payload[0]["snippet"].startswith('<resume_text source="resume.pdf"')
    assert payload[0]["snippet"].rstrip().endswith("</resume_text>")


async def test_get_candidate_profile_tool():
    executor, _ = build_executor()

    payload = json.loads(
        await executor.execute("get_candidate_profile", json.dumps({"candidate_id": "c1"}))
    )

    assert payload["id"] == "c1"


async def test_list_vacancy_applications_tool():
    executor, _ = build_executor()

    payload = json.loads(
        await executor.execute(
            "list_vacancy_applications", json.dumps({"vacancy_id": "v1", "limit": 99})
        )
    )

    assert payload[0]["vacancy_id"] == "v1"
    assert payload[0]["stage"] == "interview"


async def test_compare_tool():
    executor, _ = build_executor()

    payload = json.loads(
        await executor.execute(
            "compare_candidates",
            json.dumps({"vacancy_id": "v1", "candidate_ids": ["c1", "c2"]}),
        )
    )

    assert [item["candidate_id"] for item in payload["candidates"]] == ["c1", "c2"]


async def test_unknown_tool_returns_an_error_payload():
    executor, _ = build_executor()

    payload = json.loads(await executor.execute("delete_everything", "{}"))

    assert payload == {"error": "Unknown tool: delete_everything"}
