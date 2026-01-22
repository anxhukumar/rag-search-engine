import json
import argparse
from lib import config, hybrid_search, search_utils, augmented_generation


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")

    citations_parser = subparsers.add_parser("citations", help="Add citations in the response")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")

    question_parser = subparsers.add_parser("question", help="Ask questions from the data")
    question_parser.add_argument("question", type=str, help="question")
    question_parser.add_argument("--limit", type=int, nargs='?', default=search_utils.SEARCH_LIMIT, help="Search limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)

            rrf_data = hs.rrf_search(query, search_utils.DEFAULT_RRF_K, 5)
            llm_response = augmented_generation.augmented_generation(query, rrf_data)

            print("Search Results:")
            for key in rrf_data:
                print(f" - {rrf_data[key]["doc"]["title"]}")
            
            print("RAG Response:")
            print(llm_response)
        case "summarize":
            query = args.query

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)

            rrf_data = hs.rrf_search(query, search_utils.DEFAULT_RRF_K, args.limit)
            llm_response = augmented_generation.summarizer(query, rrf_data)

            print("Search Results:")
            for key in rrf_data:
                print(f" - {rrf_data[key]["doc"]["title"]}")
            
            print("LLM Summary:")
            print(llm_response)
        case "citations":
            query = args.query

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)

            rrf_data = hs.rrf_search(query, search_utils.DEFAULT_RRF_K, args.limit)
            llm_response = augmented_generation.citations_summarizer(query, rrf_data)

            print("Search Results:")
            for key in rrf_data:
                print(f" - {rrf_data[key]["doc"]["title"]}")
            
            print("LLM Answer:")
            print(llm_response)
        case "question":
            question = args.question

            # Load movies json
            with open(config.DATA_FILE_PATH, "r") as f:
                documents = json.load(f)["movies"]
            hs = hybrid_search.HybridSearch(documents)

            rrf_data = hs.rrf_search(question, search_utils.DEFAULT_RRF_K, args.limit)
            llm_response = augmented_generation.questions(question, rrf_data)

            print("Search Results:")
            for key in rrf_data:
                print(f" - {rrf_data[key]["doc"]["title"]}")
            
            print("Answer:")
            print(llm_response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()