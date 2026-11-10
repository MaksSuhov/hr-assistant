import json
from typing import Any

from hr_assistant.application.application_service import ApplicationService
from hr_assistant.application.candidate_service import CandidateService
from hr_assistant.application.matching_service import MatchingService
from hr_assistant.application.search_service import SearchService
from hr_assistant.core.prompt_safety import wrap_resume_text


class ToolExecutor:
    """Executes read-only tools exposed to the chat model."""

    def __init__(
        self,
        candidates: CandidateService,
        search: SearchService,
        matching: MatchingService,
        applications: ApplicationService | None = None,
    ) -> None:
        self.candidates = candidates
        self.search = search
        self.matching = matching
        self.applications = applications

    async def execute(self, name: str, arguments: str) -> str:
        args: dict[str, Any] = json.loads(arguments or "{}")
        result: Any
        payload: Any
        if name == "search_candidates":
            limit = max(1, min(int(args.get("limit", 5)), 10))
            result = await self.search.search(args["query"], limit)
            payload = [_with_wrapped_snippet(item.model_dump(mode="json")) for item in result]
        elif name == "get_candidate_profile":
            result = await self.candidates.get(args["candidate_id"])
            payload = result.model_dump(mode="json")
        elif name == "list_vacancy_applications":
            if self.applications is None:
                payload = {"error": "Vacancy applications are unavailable."}
            else:
                limit = max(1, min(int(args.get("limit", 20)), 50))
                result = await self.applications.list_for_vacancy(args["vacancy_id"], limit)
                payload = [item.model_dump(mode="json") for item in result]
        elif name == "compare_candidates":
            result = await self.matching.compare(args["vacancy_id"], args["candidate_ids"])
            payload = result.model_dump(mode="json")
        else:
            payload = {"error": f"Unknown tool: {name}"}
        return json.dumps(payload, ensure_ascii=False)


def _with_wrapped_snippet(hit: dict[str, Any]) -> dict[str, Any]:
    """Mark resume excerpts as data before the model reads a tool result."""
    snippet = hit.get("snippet")
    if isinstance(snippet, str):
        hit["snippet"] = wrap_resume_text(
            snippet,
            source=str(hit.get("source_filename", "")),
            candidate_id=str(hit.get("candidate_id", "")),
        )
    return hit
