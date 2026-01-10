#!/usr/bin/env python3
import math
import argparse
from lib.keyword_search import read_movies_data, InvertedIndex, preprocess_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index and save it to disk")


    tf_parser = subparsers.add_parser("tf", help="Returns the term frequency of a token")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Token")

    idf_parser = subparsers.add_parser("idf", help="Returns the inverse document frequency of a token")
    idf_parser.add_argument("term", type=str, help="Token")


    tfidf_parser = subparsers.add_parser("tfidf", help="Returns the term frequency-inverse document frequency of a token")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Token")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    idx = InvertedIndex()

    match args.command:
        case "search":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            print(f"Searching for: {args.query}")
            matches =read_movies_data(args.query, 5, idx)
            
            for m in matches:
                print(f"{m['id']}. {m['title']}")

        case "build":
            idx.build()
            idx.save()
        case "tf":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            print(idx.get_tf(args.doc_id, args.term))
        case "idf":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            term = preprocess_text(args.term)[0]
            idf = math.log((len(idx.docmap)+1) / (len(idx.index[term])+1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            tf = idx.get_tf(args.doc_id, args.term)

            term = preprocess_text(args.term)[0]
            idf = math.log((len(idx.docmap)+1) / (len(idx.index[term])+1))
            
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            try:
                idx.load()
            except FileNotFoundError as e:
                print("Error:", e)
                return
            bm25idf = idx.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()