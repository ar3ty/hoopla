#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Builds the inverted index and saves it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                results = search_command(args.query)
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res["title"]}")
            except Exception as e:
                print(f"{e}")
        case "build":
            build_command()
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()