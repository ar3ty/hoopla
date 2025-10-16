import string

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie["title"])
        found = False
        for title_token in preprocessed_title:
            for query_token in preprocessed_query:
                if query_token in title_token:
                    found = True
                    results.append(movie)
                    break
            if found:
                found = False
                break
        if len(results) >= limit:
            break
    return results

def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return filter(lambda x: "" != x, map(lambda x: x.strip(), text.split()))