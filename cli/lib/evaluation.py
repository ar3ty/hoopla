from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch

from .search_utils import (
    load_movies,
    load_test_cases,
    DEFAULT_SEARCH_LIMIT,
    RRF_K
)

def evaluate_command(limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    test_cases = load_test_cases()

    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hs = HybridSearch(movies)

    test_results = []
    for c in test_cases:
        query = c["query"]
        results = hs.rrf_search(query, RRF_K, limit)
        relevant_retrieved = set()
        relevant_set = set(c["relevant_docs"])

        for res in results:
            if res["title"] in relevant_set:
                relevant_retrieved.add(res["title"])
        precision = len(relevant_retrieved) / limit
        recall = len(relevant_retrieved) / len(relevant_set)
        f1 = 2 * (precision * recall) / (precision + recall)

        test_results.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": [r["title"] for r in results],
            "relevant": c["relevant_docs"],
        })
    
    return test_results