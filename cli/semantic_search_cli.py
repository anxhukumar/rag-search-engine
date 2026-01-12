#!/usr/bin/env python3
from lib import semantic_search
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()