from internal import config
import json

def read_movies_data(query: str, limit: int) -> None:
    with open(config.DATA_FILE_PATH, "r") as file:
        data = json.load(file)
    
    movies = data["movies"]
    movies.sort(key=lambda m: m["id"])
    
    res = []
    for m in movies:
        if query in m['title']:
            res.append(m)
        if len(res) == limit:
            break
    
    for i in range(len(res)):
        print(f"{i+1}. {res[i]['title']}")
