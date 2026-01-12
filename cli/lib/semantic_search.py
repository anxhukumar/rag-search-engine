from sentence_transformers import SentenceTransformer
from lib import config
import numpy as np
import json
import os

class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict] = None
        self.document_map: dict[int, dict] = {}
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("text parameter is empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        doc_str_rep = []
        for d in documents:
            self.document_map[d["id"]] = d
            doc_str_rep.append(f"{d['title']}: {d['description']}")
        self.embeddings = self.model.encode(doc_str_rep, show_progress_bar=True)
        os.makedirs(config.CACHE_FILE_PATH, exist_ok=True)
        # Write movie embeddings
        with open(config.MOVIE_EMBEDDINGS_CACHE_FILE_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for d in documents:
            self.document_map[d["id"]] = d
        if os.path.exists(config.MOVIE_EMBEDDINGS_CACHE_FILE_PATH):
            with open(config.MOVIE_EMBEDDINGS_CACHE_FILE_PATH, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)



def verify_model():
    sem = SemanticSearch()
    print(f"Model loaded: {sem.model}")
    print(f"Max sequence length: {sem.model.max_seq_length}")

def embed_text(text: str):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem = SemanticSearch()
    with open(config.DATA_FILE_PATH, "r") as f:
        documents = json.load(f)["movies"]
    embeddings = sem.load_or_create_embeddings(documents)
    print(f"Number of docs:  {len(documents)}")
    print(f"Embeddings shape:  {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
