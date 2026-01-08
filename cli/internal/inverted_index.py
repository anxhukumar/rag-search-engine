from internal import preprocess_text, config
import os
import json
import pickle

class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text.preprocess_text(text)
        for t in tokens:
            if t in self.index:
                self.index[t].add(doc_id)
            else:
                self.index[t] = set([doc_id])
    
    def get_documents(self, term: str) -> list[int]:
        token = preprocess_text.preprocess_text(term)[0]
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
        os.makedirs(os.path.dirname(config.CACHE_FILE_PATH), exist_ok=True)
        # Write index
        with open(config.INDEX_CACHE_FILE_PATH, "wb") as f:
            pickle.dump(self.index, f)
        
        # Write docmap
        with open(config.DOCMAP_CACHE_FILE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
    
    def load(self) -> None:
        try:
            # load index
            with open(config.INDEX_CACHE_FILE_PATH, "rb") as f:
                self.index = pickle.load(f)
            
            # load docmap
            with open(config.DOCMAP_CACHE_FILE_PATH, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cache file not found: {e.filename}") from e
