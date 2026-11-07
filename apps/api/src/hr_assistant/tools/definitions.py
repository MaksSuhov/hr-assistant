HR_ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": "Search resume excerpts using hybrid semantic and BM25 retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Skills or experience to find."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate_profile",
            "description": "Get a candidate card with resume sources and current status.",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}},
                "required": ["candidate_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_candidates",
            "description": "Compare candidates against every requirement in a vacancy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vacancy_id": {"type": "string"},
                    "candidate_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                    },
                },
                "required": ["vacancy_id", "candidate_ids"],
                "additionalProperties": False,
            },
        },
    },
]

