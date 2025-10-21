import os, json, string
import numpy as np
import regex as re

from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    load_movies,
    format_search_result,
    format_semantic_search_result,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    DOCUMENT_PREVIEW_LENGTH,
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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
                description=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
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
        print(f"   {res["description"]}...")
        print()

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    results = []

    i = 0
    length_words = len(words)
    while i < length_words - overlap:
        chunk = words[i : i + chunk_size]
        results.append(" ".join(chunk))
        i += chunk_size - overlap
    return results

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    results = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")

def chunk_sentences(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]
    
    sentences = [s.strip() for s in sentences if s.strip()]

    results = []
    i = 0

    while i < len(sentences):
        chunk = sentences[i : i + max_chunk_size]
        if not chunk:
            break
        results.append(" ".join(chunk))
        i += max(1, max_chunk_size - overlap)
    return results

def semantic_chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    results = chunk_sentences(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        doc_chunks = []
        chunks_metadata = []

        for i, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc

            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks_to_add = chunk_sentences(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
        
            for j, chunk in enumerate(chunks_to_add):
                doc_chunks.append(chunk)
                chunks_metadata.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunks_to_add),
                })
        self.chunk_embeddings = self.model.encode(doc_chunks)
        self.chunk_metadata = chunks_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(doc_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                loaded = json.load(f)
                self.chunk_metadata = loaded["chunks"]
            if len(self.chunk_embeddings) == loaded["total_chunks"]:
                self.documents = documents
                self.document_map = {doc["id"]: doc for doc in documents}
                return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if (
            self.chunk_embeddings is None or 
            self.chunk_embeddings.size == 0 or 
            self.chunk_metadata is None or 
            len(self.chunk_metadata) == 0
        ):
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        query_embed = self.generate_embedding(query)

        idxs_to_scores = {}
        for i, chunk_embed in enumerate(self.chunk_embeddings):
            co_sim = cosine_similarity(query_embed, chunk_embed)
            m_idx = self.chunk_metadata[i]["movie_idx"]
            if m_idx not in idxs_to_scores or idxs_to_scores[m_idx] < co_sim:
                idxs_to_scores[m_idx] = co_sim
        
        ranked = list(idxs_to_scores.items())
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in ranked[:limit]:
            doc = self.documents[idx]
            results.append(format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            ))
        
        return results

def embed_chunks():
    movies = load_movies()
    chunked_search = ChunkedSemanticSearch()
    embeddings = chunked_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    movies = load_movies()
    search_instant = ChunkedSemanticSearch()
    search_instant.load_or_create_chunk_embeddings(movies)
    results = search_instant.search_chunks(query, limit)
    print(f"Query: {query}")
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"   {res["document"][:DOCUMENT_PREVIEW_LENGTH]}...")