#!/usr/bin/env python3
from lib import semantic_search
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text",  help="Embeds a text")
    embed_text_parser.add_argument("term", type=str, help="Text")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embedquery_parser.add_argument("query", type=str, help="Query")

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.term)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embedquery":
            semantic_search.embed_query_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()