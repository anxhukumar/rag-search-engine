import argparse
import mimetypes
from google.genai import types
from lib import search_utils

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # open file
    with open(args.image, "rb") as f:
        img_data = f.read()
    
    prompt = f"""
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    
    parts = [
        prompt,
        types.Part.from_bytes(data=img_data, mime_type = mime),
        args.query.strip(),
    ]

    response = search_utils.client.models.generate_content(
        model=search_utils.MODEL, contents=parts
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens: {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()