import os
from google import genai
from google.genai import types

def ask_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=600,
        top_p=0.9,
        top_k=40
    )

    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=config
    )

    return resp.text or ""
