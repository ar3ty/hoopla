import os

from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{d['title']}: {d['description']}" for d in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: str):
        image_embed = self.embed_image(image_path)
        
        for i, embed in enumerate(self.text_embeddings):
            self.documents[i]["similarity"] = cosine_similarity(image_embed, embed)

        results = sorted(self.documents, key=lambda x: x["similarity"], reverse=True)
        return results[:5]   

    

def verify_image_embedding_command(image_path: str):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    results = ms.search_with_image(image_path)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res["title"]} (similarity: {res["similarity"]:.3f})")
        print(f"   {res["description"][:100]}\n")
