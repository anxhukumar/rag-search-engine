from sentence_transformers import CrossEncoder
from lib import search_utils, config
import time
import json

def get_response(prompt: str) -> str:
    response = search_utils.client.models.generate_content(
        model= search_utils.MODEL, contents=prompt
    )
    return response.text

def individual_rerank(query: str, doc: dict) -> int:
    
    prompt = f"""
Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.
"""
    return get_response(prompt)

def batch_rerank(query: str, docList: list[dict]) -> int:

    prompt = f"""
Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{docList}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    return get_response(prompt)

def cross_encoder_scores(pairs: list[ list[str]] ) -> list[int]:
    model = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
    scores = model.predict(pairs)
    return scores

def reranking(rrf_response: dict, query: str, method: str) -> dict:
    match method:
        case "individual":
            for key in rrf_response:
                rerank_score = individual_rerank(query, rrf_response[key]["doc"])
                rrf_response[key]["rerank_score"] = int(rerank_score.strip()) if rerank_score else 0
                time.sleep(3)
            res = dict(sorted(rrf_response.items(), key=lambda d: d[1]["rerank_score"], reverse=True))
            return res
        case "batch":
            docList = []
            for key in rrf_response:
                docList.append(rrf_response[key])
            doc_ranks_json = batch_rerank(query, docList)
            doc_ranks_list = json.loads(doc_ranks_json)
            res = {}
            for id in doc_ranks_list:
                res[id] = rrf_response[id]
            return res
        case "cross_encoder":
            pairs = []
            for key in rrf_response:
                pairs.append([query, f"{rrf_response[key]["doc"].get('title', '')} - {rrf_response[key]["doc"].get('document', '')}"])

            cross_enc_scores = cross_encoder_scores(pairs)
            for doc_key, score in zip(rrf_response, cross_enc_scores):
                rrf_response[doc_key]["cross_encoder_score"] = float(score)
            
            res = dict(sorted(rrf_response.items(), key=lambda d: d[1]["cross_encoder_score"], reverse=True))
            return res

        case _:
            return "unknown method"