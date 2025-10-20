import string, os, math

from collections import Counter, defaultdict

from nltk.stem import PorterStemmer
from pickle import dump, load

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    BM25_K1,
    BM25_B,
    load_stopwords,
    load_movies,
    format_search_result,
)

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOCLENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_and_preprocess_text(text)
        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        sum = 0
        for value in self.doc_lengths.values():
            sum += value
        return sum / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        docs = self.index.get(term, set())
        return sorted(list(docs))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_and_preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term should present only one word")
        return self.term_frequencies[doc_id][tokens[0]]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_and_preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term should present only one word")
        return math.log((len(self.docmap ) + 1) / (len(self.get_documents(tokens[0])) + 1))
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_and_preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term should present only one word")
        freq = len(self.get_documents(tokens[0]))
        return math.log((len(self.docmap ) - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            norm = 1
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * norm)
    
    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_and_preprocess_text(query)

        candidates = set()
        for token in query_tokens:
            candidates |= self.index.get(token, set())

        scores = {}
        for doc_id in candidates:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked_docs[:limit]:
            doc = self.docmap[doc_id]
            f_result = format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(f_result)
        return results

    def build(self) -> None:
        items = load_movies()
        for item in items:
            item_id = int(item["id"])
            self.docmap[item_id] = item
            self.__add_document(item_id, f"{item["title"]} {item["description"]}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            dump(self.term_frequencies, f)
        with open(DOCLENGTHS_PATH, "wb") as f:
            dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = load(f)
        with open(DOCLENGTHS_PATH, "rb") as f:
            self.doc_lengths = load(f)

def fully_matches_to_any(token: str, words: list[str]) -> bool:
    for word in words:
        if token == word:
            return True
    return False

def tokenize_and_preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = list(filter(lambda x: x != "", text.split()))
    stopwords = load_stopwords()
    tokens = list(filter(lambda x: not fully_matches_to_any(x, stopwords), tokens))
    stemmer = PorterStemmer()
    return list(map(stemmer.stem, tokens))

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_and_preprocess_text(query)

    seen, results = set(), []
    for token in query_tokens:
        indexes = idx.get_documents(token)
        for index in indexes:
            if index in seen:
                continue
            seen.add(index)
            res = idx.docmap[index]
            if not res:
                continue
            results.append(res)
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results
    return results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)