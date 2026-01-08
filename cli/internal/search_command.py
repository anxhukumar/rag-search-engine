from internal import config, preprocess_text
import json


def read_movies_data(query: str, limit: int) -> list[dict]:
    with open(config.DATA_FILE_PATH, "r") as file:
        data = json.load(file)
    
    movies = data["movies"]
    movies.sort(key=lambda m: m["id"])
    
    res = []
    query = preprocess_text.preprocess_text(query)
    for m in movies:
        title = preprocess_text.preprocess_text(m['title'])

        found_match = False
        for query_t in query:
            for title_t in title:
                if query_t in title_t:
                    found_match = True
                    break
            if found_match:
                break

        if found_match:
            res.append(m)
            if len(res) == limit:
                break
    return res