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

    idx = inverted_index.InvertedIndex()

    match args.command:
        case "search":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            print(f"Searching for: {args.query}")
            matches = search_command.read_movies_data(args.query, 5, idx)
            
            for m in matches:
                print(f"{m['id']}. {m['title']}")

        case "build":
            idx.build()
            idx.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()