import argparse
from lib import multimodal_search

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("path", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Search image")
    image_search_parser.add_argument("path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            ms = multimodal_search.verify_image_embedding(args.path)
        case "image_search":
            result = multimodal_search.image_search_command(args.path)

            for i, d in enumerate(result, start=1):
                print(f"{i}. {d['title']} (similarity: {d['similarity_score']:.3f})")
                print(d["description"])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()