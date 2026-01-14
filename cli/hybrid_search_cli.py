import argparse
from lib import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = hybrid_search.normalize(args.scores)
            for s in norm_scores:
                print(f"* {s:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()