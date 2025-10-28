import time, re, json
from sentence_transformers import CrossEncoder

from .llm_request import perform_groq_request

def parse_score(s: str) -> float | None:
    m = re.search(r"\d+(\.\d+)?", s)
    if not m:
        return None
    return max(0.0, min(10.0, float(m.group(0))))

def rerank_individual(query: str, results: list[dict], limit: int = 5) -> list:
    reranked = []
    for res in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {res.get("title", "")} - {res.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).

Give me ONLY the number, e.g. 8"""
        raw_rank = perform_groq_request(prompt)
        rank = parse_score(raw_rank)
        if not rank:
            rank = 0.0
        res["metadata"]["individual_score"] = float(rank)
        reranked.append(res)
        time.sleep(5)
    
    reranked.sort(key=lambda x: x["metadata"]["individual_score"], reverse=True)
    return reranked[:limit]

def parse_json_list(s: str) -> float | None:
    m = re.search(r"(\[.*?\])", s)
    if not m:
        return None
    return m.group(0)

def rerank_batch(query: str, results: list[dict], limit: int = 5) -> list:
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{"\n".join([f'"id": {res["id"]}, "title": {res["title"]}, "document": {res["document"][:200]}' for res in results])}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    raw_ranks = perform_groq_request(prompt)
    ranks_json = parse_json_list(raw_ranks)
    ranks = json.loads(ranks_json)
    if not ranks:
        raise Exception("ranking failed")
    
    for res in results:
        if res["id"] in ranks:
            res["metadata"]["batch_rank"] = ranks.index(res["id"]) + 1
    
    results.sort(key=lambda x: x["metadata"]["batch_rank"])
    return results[:limit]

def rerank_cross_encode(query: str, results: list[dict], limit: int = 5) -> list:
    pairs = [[query, f"{r.get('title', '')} - {r.get('document', '')}"] for r in results]
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for i, res in enumerate(results):
        res["metadata"]["cross_encoder_score"] = scores[i]
    
    results.sort(key=lambda x: x["metadata"]["cross_encoder_score"], reverse=True)
    return results[:limit]

def rerank_results(query: str, results: list[dict], method: str = "batch", limit: int = 5) -> list[dict]:
    if method == "individual":
        return rerank_individual(query, results, limit)
    elif method == "batch":
        return rerank_batch(query, results, limit)
    elif method == "cross_encoder":
        return rerank_cross_encode(query, results, limit)
    else:
        return results[:limit]