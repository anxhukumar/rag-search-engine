from lib import config, hybrid_search
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(config.GOLDEN_DATASET_FILE_PATH) as file:
            golden_data = json.load(file)
    
    with open(config.DATA_FILE_PATH) as file:
         document = json.load(file)["movies"]

    hs = hybrid_search.HybridSearch(document)
    res = []
    print(f"k={args.limit}")
    for tc in golden_data["test_cases"]:
        rrf_data = hs.rrf_search(tc["query"], 60, limit)
        retrieved_docs = []
        for k in rrf_data:
            retrieved_docs.append(rrf_data[k]["doc"]["title"])

        relevant_docs = tc["relevant_docs"]
        total_retrieved = len(retrieved_docs)
        relevant_retrieved = 0
        
        for retrived_title in retrieved_docs:
             if retrived_title in relevant_docs:
                  relevant_retrieved += 1
                       
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f"- Query: {tc['query']}\n  - Precision@{args.limit}: {precision:.4f}\n  - Recall@{args.limit}: {recall:.4f}\n  - F1 Score: {f1:.4f}\n  - Retrieved: {retrieved_docs}\n  - Relevant: {relevant_docs}")
        print("\n")

if __name__ == "__main__":
    main()