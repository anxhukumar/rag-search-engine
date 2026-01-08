from nltk.stem import PorterStemmer
from collections import Counter
from lib import config
import pickle
import string
import json
import os

class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text(text)
        for t in tokens:

            # update term freq
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][t] += 1

            if t in self.index:
                self.index[t].add(doc_id)
            else:
                self.index[t] = set([doc_id])
    
    def get_documents(self, term: str) -> list[int]:
        token = preprocess_text(term)[0]
        return sorted(self.index.get(token, []))
    
    def build(self) -> None:
        with open(config.DATA_FILE_PATH, "r") as file:
            data = json.load(file)
        movies = data["movies"]

        for m in movies:
            input_text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], input_text)
            self.docmap[m["id"]] = m
    
    def save(self) -> None:
        os.makedirs(config.CACHE_FILE_PATH, exist_ok=True)
        # Write index
        with open(config.INDEX_CACHE_FILE_PATH, "wb") as f:
            pickle.dump(self.index, f)
        
        # Write docmap
        with open(config.DOCMAP_CACHE_FILE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        # write term freq.
        with open(config.TERM_FREQ_CACHE_FILE_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self) -> None:
        try:
            # load index
            with open(config.INDEX_CACHE_FILE_PATH, "rb") as f:
                self.index = pickle.load(f)
            
            # load docmap
            with open(config.DOCMAP_CACHE_FILE_PATH, "rb") as f:
                self.docmap = pickle.load(f)

            # load term freq.
            with open(config.TERM_FREQ_CACHE_FILE_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cache file not found: {e.filename}") from e
    
    def get_tf(self, doc_id: int, term: str) -> int:
        token = preprocess_text(term)
        if len(token) > 1:
            raise Exception("More than one token found")
        
        if len(token) == 0 or doc_id not in self.term_frequencies or token[0] not in self.term_frequencies[doc_id]:
            return 0
        
        return self.term_frequencies[doc_id][token[0]]
    

stemmer = PorterStemmer()

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

def stem_tokens(tokens: list[str]) -> list[str]:
    res = []
    for t in tokens:
        res.append(stemmer.stem(t))
    return res

def preprocess_text(text: str) -> list[str]:
    return stem_tokens(remove_stop_words(tokenize(remove_punctuation(lower_case(text)))))


def read_movies_data(query: str, limit: int, idx: InvertedIndex) -> list[dict]:
    
    res = []

    query = preprocess_text(query)
    for token in query:
        matching_docs = idx.get_documents(token)
        for id in matching_docs:
            if id in idx.docmap:
                res.append(idx.docmap[id])
                if len(res) == limit:
                    return res