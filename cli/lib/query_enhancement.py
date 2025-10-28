from .llm_request import perform_groq_request

def enhance_spell(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Return only the corrected query text, no quotes, no prefix, same lettercase."""
    resp = perform_groq_request(prompt)
    return resp if resp else query

def enhance_rewrite(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Return only the corrected query text, no quotes, no prefix."""
    resp = perform_groq_request(prompt)
    return resp if resp else query

def enhance_expand(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"

Return only the corrected query text, no quotes, no prefix.
"""
    resp = perform_groq_request(prompt)
    return resp if resp else query

def enhance_query(query: str, enhance: str) -> str:
    match enhance:
        case "spell":
            return enhance_spell(query)
        case "rewrite":
            return enhance_rewrite(query)
        case "expand":
            return enhance_expand(query)
        case _:
            print("Unknown enhancement. Performing regular search.")
            return query