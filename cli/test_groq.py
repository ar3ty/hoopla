import os
from dotenv import load_dotenv
from groq import Groq

def main():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    print(f"Using key <hided>...")

    client = Groq(api_key=api_key)

    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    resp = client.chat.completions.create(
        model="groq/compound",
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )

    text = resp.choices[0].message.content
    print(text)

    usage = resp.usage
    if usage:
        print(f"Prompt Tokens: {usage.prompt_tokens}")
        print(f"Response Tokens: {usage.completion_tokens}")


if __name__ == "__main__":
    main()