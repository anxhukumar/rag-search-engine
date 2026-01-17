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

def spell(query: str) -> str:

    prompt = f"""
Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Always return the query only and keep the casing same as it was before.
"""

    return get_response(prompt)

def rewrite(query: str) -> str:

    prompt = f"""
Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Always return the query only.
"""
    
    return get_response(prompt)

def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return spell(query)
        case "rewrite":
            return rewrite(query)
        case _:
            return "Unknown method"