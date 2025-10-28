import argparse

from lib.hybrid_search import (
    normalize,
    weighted_search,
    rrf_search,
)

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    RRF_K,
) 

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalizer = subparsers.add_parser("normalize", help="Normalize given list of numbers")
    normalizer.add_argument("list", nargs='+', type= float, help="List of numbers to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA, help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned resources (default=5)")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform Reciprocal Rank Fusion search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("--k", type=float, nargs='?', default=60, help="Weight parameter for RRF (default=60)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="LLM reranks results")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned resources (default=5)")

    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.list)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.enhance, args.rerank_method, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()