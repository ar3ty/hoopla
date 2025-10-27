import os
from typing import Optional

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    DEFAULT_ALPHA,
    RRF_K,
    load_movies,
    format_search_result,
)
from .query_enhancement import enhance_query

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
        bm_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        bm_scores = [d["score"] for d in bm_results]
        sem_scores = [d["score"] for d in sem_results]
        norm_bms = normalize_scores(bm_scores)
        norm_sems = normalize_scores(sem_scores)

        id_to_docs_n_scores = {}
        for i, bm in enumerate(norm_bms):
            doc_dict = bm_results[i]
            doc_id = doc_dict["id"]
            if not id_to_docs_n_scores.get(doc_id): 
                id_to_docs_n_scores[doc_id] = {
                    "title": doc_dict["title"],  
                    "document": doc_dict["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0
                }
            if id_to_docs_n_scores[doc_id]["bm25_score"] < bm:
                id_to_docs_n_scores[doc_id]["bm25_score"] = bm
            
        for i, sem in enumerate(norm_sems):
            doc_dict = sem_results[i]
            doc_id = doc_dict["id"]
            if not id_to_docs_n_scores.get(doc_id): 
                id_to_docs_n_scores[doc_id] = {
                    "title": doc_dict["title"],  
                    "document": doc_dict["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0
                }
            if id_to_docs_n_scores[doc_id]["semantic_score"] < sem:
                id_to_docs_n_scores[doc_id]["semantic_score"] = sem

        results = []    
        for k, v in id_to_docs_n_scores.items():
            hs = hybrid_score(v["bm25_score"], v["semantic_score"], alpha)
            results.append(format_search_result(
                doc_id=k,
                title=v["title"],
                document=v["document"],
                score=hs,
                bm25_score=v["bm25_score"],
                semantic_score=v["semantic_score"],
            ))
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]
    
    def rrf_search(self, query, k=RRF_K, limit=DEFAULT_SEARCH_LIMIT):
        bm_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)

        id_to_docs_n_ranks = {}
        for i, bm in enumerate(bm_results, 1):
            doc_dict = bm
            doc_id = doc_dict["id"]
            if not id_to_docs_n_ranks.get(doc_id): 
                id_to_docs_n_ranks[doc_id] = {
                    "title": doc_dict["title"],  
                    "document": doc_dict["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
            id_to_docs_n_ranks[doc_id]["bm25_rank"] = i
            id_to_docs_n_ranks[doc_id]["rrf_score"] += rrf_score(i, k)

        for i, sem in enumerate(sem_results, 1):
            doc_dict = sem
            doc_id = doc_dict["id"]
            if not id_to_docs_n_ranks.get(doc_id): 
                id_to_docs_n_ranks[doc_id] = {
                    "title": doc_dict["title"],  
                    "document": doc_dict["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
            id_to_docs_n_ranks[doc_id]["semantic_rank"] = i
            id_to_docs_n_ranks[doc_id]["rrf_score"] += rrf_score(i, k)

        results = []    
        for k, v in id_to_docs_n_ranks.items():
            results.append(format_search_result(
                doc_id=k,
                title=v["title"],
                document=v["document"],
                score=v["rrf_score"],
                bm25_rank=v["bm25_rank"],
                semantic_rank=v["semantic_rank"],
            ))
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]
    
def normalize_scores(scores: list) -> list:
    if not scores:
        return []
    
    min_x, max_x = min(scores), max(scores)
    if min_x == max_x:
        return [1.0] * len(scores)
    
    results = []
    for s in scores:
        results.append((s - min_x) / (max_x - min_x))
    return results

def normalize(lst: list) -> None:
    scores = normalize_scores(lst)
    for s in scores:
        print(f"* {s:.4f}")

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    print(f"Weighted Hybrid Search Results for '{query}' (alpha={alpha})")
    print(f"Alpha {alpha}: {int(alpha * 100)}% Keyword, {int((1 - alpha) * 100)}% Semantic")
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']}")
        print(f"   Hybrid Score: {res['score']:.3f}")
        print(f"   BM25: {res['metadata']['bm25_score']:.3f}, Semantic: {res['metadata']['semantic_score']:.3f}")
        print(f"   {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")

def rrf_score(rank, k=RRF_K):
    return 1 / (k + rank)

def rrf_search(query: str, k: int = RRF_K, enhance: Optional[str] = None, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    if enhance:
        enhanced_query = enhance_query(query, enhance)
        print( f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query

    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k, limit)
    print(f"RRF Hybrid Search Results for '{query}' (k={k})")
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']}")
        print(f"   RRF Score: {res['score']:.3f}")

        metadata = res.get("metadata", {})
        ranks = []
        if res["metadata"].get("bm25_rank"):
            ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
        if res["metadata"].get("semantic_rank"):
            ranks.append(f"Semaintic Rank: {metadata['semantic_rank']}")
        
        print(f"   {", ".join(ranks)}")
        print(f"   {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")