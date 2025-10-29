import os, base64
from dotenv import load_dotenv
from groq import Groq

import mimetypes

def describe_image(image: str, query: str):
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    with open(image, "rb") as f:
        data = f.read()
    base64_image = base64.b64encode(data).decode('utf-8')

    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Use correct proper nouns when identifiable
- Return only the rewritten query, without any additional commentary"""

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    cli = Groq(api_key=api_key)
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    resp = cli.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{base64_image}",
                    }
                }
            ],       
        }],
    )
    text = (resp.choices[0].message.content or "").strip().strip('"')

    return text, resp.usage.total_tokens
    

    