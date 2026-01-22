from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib import config
import json

class MultimodalSearch:
    
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        img = Image.open(image_path)
        embedding = self.model.encode([img], show_progress_bar=True)
        return embedding[0]
    
    def search_with_image(self, image_path):
        img = Image.open(image_path)
        image_embed = self.model.encode([img], show_progress_bar=True)

        doc_with_score = []
        for i, text_embed in enumerate(self.text_embeddings):
            cos_sim = cosine_similarity(image_embed, text_embed).item()
            doc_with_score.append(
                {
                    "doc_id": i,
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "similarity_score": cos_sim,
                }
            )
        # Sort the docs and return 5

        sorted_docs = sorted(doc_with_score, key=lambda d: d["similarity_score"], reverse=True)
        return sorted_docs[:5]

def verify_image_embedding(image_path):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    with open(config.DATA_FILE_PATH, "r") as f:
        documents = json.load(f)["movies"]
    
    ms = MultimodalSearch(documents)
    return ms.search_with_image(image_path)