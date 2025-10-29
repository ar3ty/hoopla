from .llm_request import perform_groq_request
from .semantic_search import SemanticSearch
from .hybrid_search import HybridSearch

from .search_utils import (
    load_movies,
    RRF_K,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER
)

def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple:
    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hs = HybridSearch(movies)

    search_results = hs.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    prompt = f"""You are a RAG agent that provides a human answer
to the user's query based on the documents that were retrieved during search.
Answer the question or provide information based on the provided documents. 
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Provide a comprehensive answer that addresses the query.

Query: {query}

Documents:
{"\n".join([f"{i}: title - {r['title']}, document - {r['document'][:100]}" for i, r in enumerate(search_results[:limit], 1)])}"""
    
    response = perform_groq_request(prompt).strip()
    results = [r["title"] for r in search_results[:limit]]
    return results, response

def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple:
    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hs = HybridSearch(movies)

    search_results = hs.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{"\n".join([f"{i}: title - {r['title']}, document - {r['document'][:100]}" for i, r in enumerate(search_results[:limit], 1)])}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources.
"""
    
    response = perform_groq_request(prompt).strip()
    results = [r["title"] for r in search_results[:limit]]
    return results, response

def citations_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple:
    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hs = HybridSearch(movies)

    search_results = hs.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{"\n".join([f"{i}: title - {r['title']}, document - {r['document'][:100]}" for i, r in enumerate(search_results[:limit], 1)])}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    
    response = perform_groq_request(prompt).strip()
    results = [r["title"] for r in search_results[:limit]]
    return results, response

def question_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple:
    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hs = HybridSearch(movies)

    search_results = hs.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{"\n".join([f"{i}: title - {r['title']}, document - {r['document']}" for i, r in enumerate(search_results[:limit], 1)])}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""
    
    response = perform_groq_request(prompt).strip()
    results = [r["title"] for r in search_results[:limit]]
    return results, response