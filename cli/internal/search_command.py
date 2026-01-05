from internal import config
import json

def read_movies_data(query: str, limit: int) -> list[str]:
    with open(config.DATA_FILE_PATH, "r") as file:
        data = json.load(file)
    
    movies = data["movies"]
    movies.sort(key=lambda m: m["id"])
    
    res = []
    for m in movies:
        # lowercase before matching
        if query.lower() in m['title'].lower():
            res.append(m)
        if len(res) == limit:
            break
    return res
