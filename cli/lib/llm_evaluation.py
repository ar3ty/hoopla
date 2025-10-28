import json

from .llm_request import perform_groq_request
from .reranking import parse_json_list

def evaluate_rrf_results(query: str, rrf_results: list[dict]) -> list:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join([f'"id": {res["id"]}, "title": {res["title"]}, "document": {res["document"][:200]}' for res in rrf_results])}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    raw_eval = perform_groq_request(prompt)
    eval_json = parse_json_list(raw_eval)
    eval = json.loads(eval_json)
    if not eval:
        raise Exception("evaluation failed")

    return [{"title": r["title"], "eval": eval[i]} for i, r in enumerate(rrf_results)]