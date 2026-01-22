from lib import search_utils

def get_response(prompt: str) -> str:
    response = search_utils.client.models.generate_content(
        model= search_utils.MODEL, contents=prompt
    )
    return response.text

def augmented_generation(query: str, results: dict) -> str:
    formatted_results = []
    for key in results:
        formatted_results.append(f"{results[key]['doc']['title']} - {results[key]['doc']['description']}")
    
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{chr(10).join(formatted_results)}

Provide a comprehensive answer that addresses the query:"""

    return get_response(prompt)

def summarizer(query: str, results: dict) -> str:
    formatted_results = []
    for key in results:
        formatted_results.append(f"{results[key]['doc']['title']} - {results[key]['doc']['description']}")

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{chr(10).join(formatted_results)}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    return get_response(prompt)

def citations_summarizer(query: str, results: dict) -> str:
    formatted_results = []
    for key in results:
        formatted_results.append(f"{results[key]['doc']['title']} - {results[key]['doc']['description']}")

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{chr(10).join(formatted_results)}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    
    return get_response(prompt)

def questions(question: str, results: dict) -> str:
    formatted_results = []
    for key in results:
        formatted_results.append(f"{results[key]['doc']['title']} - {results[key]['doc']['description']}")
    
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{chr(10).join(formatted_results)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    
    return get_response(prompt)