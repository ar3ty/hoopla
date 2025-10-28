import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
cli = Groq(api_key=api_key)
model = "groq/compound"

def perform_groq_request(prompt: str) -> str:
    resp = cli.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,       
        }]
    )
    text = (resp.choices[0].message.content or "").strip().strip('"')
    return text