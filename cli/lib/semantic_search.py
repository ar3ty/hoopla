import os
import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    load_movies,
    format_semantic_search_result,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents: list[dict]) -> list:
        self.documents = documents
        self.document_map = {}
        doc_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc["title"]} {doc["description"]}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
        
    def load_or_create_embeddings(self, documents: list[dict]) -> list:
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                self.documents = documents
                self.document_map = {doc["id"]: doc for doc in documents}
                return self.embeddings
        
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if (
            self.embeddings is None or 
            self.embeddings.size == 0 or 
            self.documents is None or 
            len(self.documents) == 0
        ):
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)

        scores = []
        for i, doc_embed in enumerate(self.embeddings):
            scores.append((cosine_similarity(query_embedding, doc_embed), self.documents[i]))
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:limit]:
            f_result = format_semantic_search_result(
                score=score,
                title=doc["title"],
                description=doc["description"],
            )
            results.append(f_result)
        return results

def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    movies = load_movies()
    embeds = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeds.shape[0]} vectors in {embeds.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()
    for i, res in enumerate(results, 1):
        print(f"{i}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"   {res["description"][:100]}...")
        print()

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    results = []

    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        results.append(" ".join(chunk))
        i += chunk_size
    return results

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    results = fixed_size_chunking(text, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")