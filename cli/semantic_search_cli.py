#!/usr/bin/env python3
from lib import semantic_search, search_utils, config
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text",  help="Embeds a text")
    embed_text_parser.add_argument("term", type=str, help="Text")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embedquery_parser.add_argument("query", type=str, help="Query")

    search_parser = subparsers.add_parser("search", help="Search movies using vector cosine similarity")
    search_parser.add_argument("query", type=str, help="Query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Limit the search results")

    chunk_parser = subparsers.add_parser("chunk", help="Splits text into chunks")
    chunk_parser.add_argument("text", type=str, help="Text")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=search_utils.TEXT_CHUNK_SIZE, help="Chunk size limit")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=search_utils.TEXT_CHUNK_OVERLAP, help="Add overlap to text chunks")

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
        case "search":
            sem = semantic_search.SemanticSearch()

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            _ = sem.load_or_create_embeddings(documents)

            matches = sem.search(args.query, args.limit)
            for i, m in enumerate(matches, start=1):
                print(f"{i}. {m['title']} (score: {m['score']:.4f})")
                print(f"{m['description']}")
        case "chunk":
            text = args.text
            text_arr = text.split()
            final_chunks = []
            i = 0
            while i+args.overlap < len(text_arr):
                chunk = text_arr[i:i+args.chunk_size]
                final_chunks.append(" ".join(chunk))
                i = (i+args.chunk_size) - args.overlap
            print(f"Chunking {len(text)} characters")
            for i, s in enumerate(final_chunks, start=1):
                print(f"{i}. {s}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()