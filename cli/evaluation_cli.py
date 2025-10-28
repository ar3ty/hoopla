import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from cli.lib.evaluation import evaluate_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit
    
    results = evaluate_command(limit)

    print(f"k={limit}\n")

    for r in results:
        print(f"- Query: {r['query']}")
        print(f"  - Precision@{limit}: {r['precision']:.4f}")
        print(f"  - Recall@{limit}: {r['recall']:.4f}")
        print(f"  - F1 Score: {r['f1']:.4f}")
        print(f"  - Retrieved: {', '.join(r['retrieved'])}")
        print(f"  - Relevant: {', '.join(r['relevant'])}\n")

if __name__ == "__main__":
    main()