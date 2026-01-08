from internal import config, preprocess_text, inverted_index
import json


def read_movies_data(query: str, limit: int, idx: inverted_index.InvertedIndex) -> list[dict]:
    
    res = []

    query = preprocess_text.preprocess_text(query)
    for token in query:
        matching_docs = idx.get_documents(token)
        for id in matching_docs:
            if id in idx.docmap:
                res.append(idx.docmap[id])
                if len(res) == limit:
                    return res