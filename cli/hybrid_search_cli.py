import argparse
import json
from lib import hybrid_search, search_utils, config, enhance_search, reranking, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Get combined weighted score")
    weighted_search_parser.add_argument("query", type=str, help="Query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=search_utils.ALPHA_VAL, help="Alpha parameter to determine seach type weightage")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")

    rrf_search_parser = subparsers.add_parser("rrf-search", help = "Perform Reciprocal Rank Fusion search")
    rrf_search_parser.add_argument("query", type=str, help="Query")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=search_utils.DEFAULT_RRF_K, help="Controls the gap between ranks")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank the results")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="evaluate search with LLM")

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
        case "rrf-search":
            # LOGS
            print(f"Original query: {args.query}")

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)

            if args.enhance:
                method = args.enhance
                old_args_query = args.query
                args.query = enhance_search.enhance_query(args.query, method)
                print(f"Enhanced query ({method}): '{old_args_query}' -> '{args.query}'")
            
            # Check if if re-rank is set to individual
            if args.rerank_method:
                extra_limit = args.limit * 5
            else:
                extra_limit = args.limit
            
            final_results = hs.rrf_search(args.query, args.k, extra_limit) #RRF DATA

            # LOG RRF results
            print(f"RRF search returned {len(final_results)} results")
            print("Top 10 from RRF:")
            for i, key in enumerate(list(final_results.keys())[:10], start=1):
                print(f"{i}. {final_results[key]["doc"]["title"]} (RRF Score: {final_results[key]["total_rrf_score"]:.4f})")

            
            if args.rerank_method:
                final_results = reranking.reranking(final_results, args.query, args.rerank_method)

                # LOG Reranked results
                print(f"After reranking with {args.rerank_method}:")
                print("Top 10 after reranking:")
                for i, key in enumerate(list(final_results.keys())[:10], start=1):
                    print(f"{i}. {final_results[key]["doc"]["title"]}")
            

            # LLM EVALUATION
            print(f"LLM EVALUATION:")
            llm_eval_scores = evaluation.evaluate_query_results(args.query, final_results)
            for i, key in enumerate(final_results):
                print(f"{i+1}. {final_results[key]["doc"]["title"]}: {llm_eval_scores[i]}/3")

            if args.rerank_method == "cross_encoder":
                for i, key in enumerate(final_results, start=1):
                    print(f"{i}. {final_results[key]["doc"]["title"]}\nCross Encoder Score: {final_results[key]["cross_encoder_score"]}\nRRF Score: {final_results[key]["total_rrf_score"]}\nBM25 Rank: {final_results[key]["bm25_rank"]}, Semantic Rank: {final_results[key]["semantic_rank"]}\n{final_results[key]["doc"]["description"]}")
                    if i == args.limit:
                        break
            else:
                for i, key in enumerate(final_results, start=1):
                    print(f"{i}. {final_results[key]["doc"]["title"]}\nRerank Rank: {i}\nRRF Score: {final_results[key]["total_rrf_score"]}\nBM25 Rank: {final_results[key]["bm25_rank"]}, Semantic Rank: {final_results[key]["semantic_rank"]}\n{final_results[key]["doc"]["description"]}")
                    if i == args.limit:
                        break
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()