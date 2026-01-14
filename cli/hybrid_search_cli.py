import argparse
import json
from lib import hybrid_search, search_utils, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Get combined weighted score")
    weighted_search_parser.add_argument("query", type=str, help="Query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=search_utils.ALPHA_VAL, help="Alpha parameter to determine seach type weightage")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = hybrid_search.normalize(args.scores)
            for s in norm_scores:
                print(f"* {s:.4f}")
        case "weighted-search":
            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)
            data: dict = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, k in enumerate(data, start=1):
                print(f"{i}. {data[k]['doc']['title']}\nHybrid Score: {data[k]['hybrid_score']}\nBM25: {data[k]['keyword_score']}, Semantic: {data[k]['semantic_score']}\n{data[k]['doc']['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()