from lib import search_utils
import json

def get_response(prompt: str) -> str:
    response = search_utils.client.models.generate_content(
        model= search_utils.MODEL, contents=prompt
    )
    return response.text

def evaluate_query_results(query: str, results: dict) -> list[int]:
    formatted_results = []
    for key in results:
        formatted_results.append(f"{results[key]['doc']['title']} - {results[key]['doc']['description'][:200]}")
    
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    
    return json.loads(get_response(prompt))