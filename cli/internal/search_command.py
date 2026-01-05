from internal import config
import string
import json


def read_movies_data(query: str, limit: int) -> list[dict]:
    with open(config.DATA_FILE_PATH, "r") as file:
        data = json.load(file)
    
    movies = data["movies"]
    movies.sort(key=lambda m: m["id"])
    
    res = []
    query = preprocess_text(query)
    for m in movies:
        title = preprocess_text(m['title'])

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

def lower_case(text: str) -> str:
    return text.lower()
    
def remove_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def tokenize(text: str) -> list[str]:
    return text.split()

def remove_stop_words(tokens: list[str]) -> list[str]:
    res = []
    with open(config.STOP_WORDS_FILE_PATH, "r") as file:
        stop_words = file.read()
    stop_words_set = set(stop_words.splitlines())
    for t in tokens:
        if t not in stop_words_set:
            res.append(t)
    return res

def preprocess_text(text: str) -> list[str]:
    return remove_stop_words(tokenize(remove_punctuation(lower_case(text))))