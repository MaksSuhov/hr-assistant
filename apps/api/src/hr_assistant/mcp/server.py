from fastmcp import FastMCP

from hr_assistant.container import ServiceContainer


def create_mcp_server(container: ServiceContainer) -> FastMCP:
    mcp = FastMCP("HR Assistant Tools")

    @mcp.tool
    async def search_candidates(query: str, limit: int = 5) -> list[dict]:
        """Search resume excerpts with hybrid semantic and BM25 retrieval."""
        results = await container.search.search(query, min(max(limit, 1), 10))
        return [item.model_dump(mode="json") for item in results]

    @mcp.tool
    async def get_candidate_profile(candidate_id: str) -> dict:
        """Return one candidate card, its resume sources and applications."""
        result = await container.candidates.get(candidate_id)
        return result.model_dump(mode="json")

    @mcp.tool
    async def list_vacancy_applications(vacancy_id: str, limit: int = 20) -> list[dict]:
        """List candidates who applied to a vacancy and their current stage."""
        results = await container.applications.list_for_vacancy(vacancy_id, min(max(limit, 1), 50))
        return [item.model_dump(mode="json") for item in results]

    @mcp.tool
    async def compare_candidates(vacancy_id: str, candidate_ids: list[str]) -> dict:
        """Compare candidates against vacancy requirements and surface evidence or questions."""
        if len(candidate_ids) < 2 or len(candidate_ids) > 10:
            raise ValueError("Compare between two and ten candidates.")
        result = await container.matching.compare(vacancy_id, candidate_ids)
        return result.model_dump(mode="json")

    return mcp
