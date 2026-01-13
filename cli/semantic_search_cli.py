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

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks text in a way that preserves meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="Text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=search_utils.MAX_CHUNK_SIZE, help="Maximum chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=search_utils.TEXT_CHUNK_OVERLAP, help="Add overlap to text chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embeds text chunks")


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
            final_chunks = semantic_search.chunk_command(text, args.overlap, args.chunk_size)
            print(f"Chunking {len(text)} characters")
            for i, s in enumerate(final_chunks, start=1):
                print(f"{i}. {s}")
        case "semantic_chunk":
            final_chunks = semantic_search.semantic_chunk(args.text, args.overlap, args.max_chunk_size)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, s in enumerate(final_chunks, start=1):
                print(f"{i}. {s}")
        case "embed_chunks":
            csem = semantic_search.ChunkedSemanticSearch()

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            _ = csem.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(csem.chunk_embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()