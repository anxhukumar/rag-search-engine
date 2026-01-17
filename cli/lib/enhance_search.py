import os
from dotenv import load_dotenv
from google import genai

MODEL = "gemini-2.5-flash"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def get_response(prompt: str) -> str:
    response = client.models.generate_content(
        model= MODEL, contents=prompt
    )
    return response.text

def enhance_spell(query: str) -> str:

    prompt = f"""
Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Always return the query only and keep the casing same as it was before.
"""

    return get_response(prompt)

def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return enhance_spell(query)
        case _:
            return "Unknown method"