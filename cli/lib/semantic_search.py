from sentence_transformers import SentenceTransformer
from lib import config
import numpy as np
import json
import os
import re

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
    
    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
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
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_emb = self.generate_embedding(query)
        similarity: list[tuple[float, dict]] = []
        
        for i in range(len(self.embeddings)):
            co_sim = cosine_similarity(query_emb, self.embeddings[i])
            similarity.append((co_sim, self.documents[i]))
        
        sorted_similarity = sorted(similarity, key=lambda sim: sim[0], reverse=True)

        final_res: list[dict] = []
        for i in range(min(limit, len(sorted_similarity))):
            final_res.append(
                {
                    "score": sorted_similarity[i][0],
                    "title": sorted_similarity[i][1]["title"],
                    "description": sorted_similarity[i][1]["description"],
                }
            )
        return final_res


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        chunks: list[str] = []
        chunk_meta: list[dict] = []
        for movie_idx, d in enumerate(documents):
            if not d['description']:
                continue
            semantic_chunks = semantic_chunk(d['description'], 1, 4)
            for chunk_idx, chunk_text in enumerate(semantic_chunks):
                chunks.append(chunk_text)
                chunk_meta.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(semantic_chunks),
                    }
                )
            self.document_map[d["id"]] = d
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta
        os.makedirs(config.CACHE_FILE_PATH, exist_ok=True)
        # Write chunk embeddngs
        with open(config.CHUNK_EMBEDDINGS_CACHE_FILE_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)
        # Write chunk metadata
        with open(config.CHUNK_METADATA_CACHE_FILE_PATH, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(chunks)
                },
                f,
                indent=2
                )
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for d in documents:
            self.document_map[d["id"]] = d
        if os.path.exists(config.CHUNK_EMBEDDINGS_CACHE_FILE_PATH) and os.path.exists(config.CHUNK_METADATA_CACHE_FILE_PATH):
            with open(config.CHUNK_EMBEDDINGS_CACHE_FILE_PATH, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(config.CHUNK_METADATA_CACHE_FILE_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_emb = self.generate_embedding(query)
        chunk_score = []

        for chunk_idx, v in enumerate(self.chunk_embeddings):
            cos_sim = cosine_similarity(v, query_emb)
            chunk_score.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                    "score": cos_sim 
                }
            )
        movie_idx_scores = {}
        for i in range(len(chunk_score)):
            if chunk_score[i]["movie_idx"] not in movie_idx_scores or chunk_score[i]["score"] > movie_idx_scores[chunk_score[i]["movie_idx"]]:
                movie_idx_scores[chunk_score[i]["movie_idx"]] = chunk_score[i]["score"]
                

        sorted_movie_idx_scores = sorted(movie_idx_scores.items(), key=lambda item: item[1], reverse=True)[:limit]

        res = []
        for v in sorted_movie_idx_scores:
            obj = self.documents[v[0]]
            res.append(
                {
                    "id": obj["id"],
                    "title": obj["title"],
                    "document": obj["description"][:100],
                    "score": round(v[1], 2),
                    "metadata": {}
                }
            )
        return res
    

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

def embed_query_text(query: str):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_command(text: str, overlap: int, chunk_size: int) -> list[str]:
    res = []
    text_arr = text.split()
    i = 0
    while i + overlap < len(text_arr):
        chunk = text_arr[i:i+chunk_size]
        res.append(" ".join(chunk))
        i = (i+chunk_size) - overlap
    return res

def semantic_chunk(text: str, overlap: int, max_chunk_size: int) -> list[str]:
    res = []
    text = text.strip()
    if text == "":
        return []

    text_arr = re.split(r"(?<=[.!?])\s+", text)
    if len(text_arr) == 1:
        if not text_arr[0].endswith((".", "!", "?")):
            return [text]

    i = 0
    while i + overlap < len(text_arr):
        chunk = text_arr[i:i + max_chunk_size]
        chunk_text = " ".join(chunk).strip()
        if chunk_text != "":
            res.append(chunk_text)
        i = (i+max_chunk_size) - overlap
    return res