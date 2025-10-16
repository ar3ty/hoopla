import string

from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_and_preprocess_text(query)
        title_tokens = tokenize_and_preprocess_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
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
