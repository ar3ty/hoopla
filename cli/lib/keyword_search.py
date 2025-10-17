import string, os

from nltk.stem import PorterStemmer
from pickle import dump, load

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    PROJECT_ROOT,
    load_stopwords,
    load_movies,
)

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {str: set}
        self.docmap = {int: dict}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_and_preprocess_text(text)
        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        docs = self.index.get(term, set())
        return sorted(list(docs))

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

    def load(self) -> None:
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_and_preprocess_text(query)

    results = []
    for token in query_tokens:
        indexes = idx.get_documents(token)
        for index in indexes:
            results.append(idx.docmap[index])
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results
    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for title_token in title_tokens:
        for query_token in query_tokens:
            if query_token in title_token:
                return True
    return False

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
