#!/usr/bin/env python3

import argparse
from internal import search_command, inverted_index

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index and save it to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matches = search_command.read_movies_data(args.query, 5)
            
            for i, m in enumerate(matches):
                print(f"{i+1}. {m['title']}")
        case "build":
            idx = inverted_index.InvertedIndex()
            idx.build()
            idx.save()
            docs = idx.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()