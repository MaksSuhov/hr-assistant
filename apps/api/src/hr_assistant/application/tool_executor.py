import json
from typing import Any

from hr_assistant.application.candidate_service import CandidateService
from hr_assistant.application.matching_service import MatchingService
from hr_assistant.application.search_service import SearchService


class ToolExecutor:
    """Executes read-only tools exposed to the chat model."""

    def __init__(
        self,
        candidates: CandidateService,
        search: SearchService,
        matching: MatchingService,
    ) -> None:
        self.candidates = candidates
        self.search = search
        self.matching = matching

    async def execute(self, name: str, arguments: str) -> str:
        args: dict[str, Any] = json.loads(arguments or "{}")
        if name == "search_candidates":
            limit = max(1, min(int(args.get("limit", 5)), 10))
            result = await self.search.search(args["query"], limit)
            payload = [item.model_dump(mode="json") for item in result]
        elif name == "get_candidate_profile":
            result = await self.candidates.get(args["candidate_id"])
            payload = result.model_dump(mode="json")
        elif name == "compare_candidates":
            result = await self.matching.compare(args["vacancy_id"], args["candidate_ids"])
            payload = result.model_dump(mode="json")
        else:
            payload = {"error": f"Unknown tool: {name}"}
        return json.dumps(payload, ensure_ascii=False)

