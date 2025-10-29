import argparse
from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command,
)

from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Generate multi-document summary")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned resources (default=5)")

    citations_parser = subparsers.add_parser("citations", help="Generate citations-aware answer")
    citations_parser.add_argument("query", type=str, help="Search query for answer")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned resources (default=5)")

    question_parser = subparsers.add_parser("question", help="Generate RAG answer")
    question_parser.add_argument("query", type=str, help="Search query for answer")
    question_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Number of returned resources (default=5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            results, response = rag_command(args.query)
            print("Search Results:")
            for r in results:
                print(f"  - {r}")
            print("\nRAG Response:")
            print(response)
        case "summarize":
            results, response = summarize_command(args.query, args.limit)
            print("Search Results:")
            for r in results:
                print(f"  - {r}")
            print("\nLLM Summary:")
            print(response)
        case "citations":
            results, response = citations_command(args.query, args.limit)
            print("Search Results:")
            for r in results:
                print(f"  - {r}")
            print("\nLLM Answer:")
            print(response)
        case "question":
            results, response = question_command(args.query, args.limit)
            print("Search Results:")
            for r in results:
                print(f"  - {r}")
            print("\nAnswer:")
            print(response)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()