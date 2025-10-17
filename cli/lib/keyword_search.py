import string, os, collections

from nltk.stem import PorterStemmer
from pickle import dump, load

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    load_stopwords,
    load_movies,
)

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {str: set}
        self.docmap = {int: dict}
        self.term_frequencies = {int: collections.Counter}

    def __add_document(self, doc_id: str, text: str) -> None:
        tokens = tokenize_and_preprocess_text(text)
        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        docs = self.index.get(term, set())
        return sorted(list(docs))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_and_preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term should present only one word")
        return self.term_frequencies[doc_id][term]

    def build(self) -> None:
        items = load_movies()
        for item in items:
            item_id = int(item["id"])
            self.docmap[item_id] = item
            self.term_frequencies[item_id] = collections.Counter()
            self.__add_document(item_id, f"{item["title"]} {item["description"]}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: str, term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(int(doc_id), term)

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
