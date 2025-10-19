#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
)

from lib.search_utils import (
    BM25_K1,
    BM25_B,
    DEFAULT_SEARCH_LIMIT,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index and saves it to disk")

    tf_parser = subparsers.add_parser("tf", help="Prints the term frequency in the document with the given ID.")
    tf_parser.add_argument("doc_id", type=int, help="Document to look into")
    tf_parser.add_argument("term", type=str, help="Term to look frequency for")

    idf_parser = subparsers.add_parser("idf", help="Prints the inverse document frequency in the database.")
    idf_parser.add_argument("term", type=str, help="Term to look inverse frequency for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Prints the TF-IDF for the document with the given ID.")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document to look into")
    tf_idf_parser.add_argument("term", type=str, help="Term to look frequency for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Prints the BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Prints the BM25 TF score for a given term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document to look into")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of returned sources")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                results = search_command(args.query)
                print("Found:")
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res["title"]}")
            except Exception as e:
                print(f"{e}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            try:
                frequency = tf_command(args.doc_id, args.term)
                print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {frequency}")
            except Exception as e:
                print(f"{e}")
        case "idf":
            try:
                idf = idf_command(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except Exception as e:
                print(f"{e}")
        case "tfidf":
            try:
                tf_idf = tf_idf_command(args.doc_id, args.term)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except Exception as e:
                print(f"{e}")
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except Exception as e:
                print(f"{e}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except Exception as e:
                print(f"{e}")
        case "bm25search":
            print(f"Searching for: {args.query}")
            try:
                results = bm25search_command(args.query, args.limit)
                print("Found:")
                for i, res in enumerate(results, 1):
                    print(f"{i}. ({res['id']}) {res['title']} - Score {res['score']:.2f}")
            except Exception as e:
                print(f"{e}")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()