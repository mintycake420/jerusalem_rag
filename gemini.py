from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai

# --- Load API.env explicitly (NOT .env) ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# --- Safety check ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        f"GEMINI_API_KEY not found. Expected it in {ENV_PATH}"
    )

# --- Create client explicitly with key ---
client = genai.Client(api_key=api_key)

# --- Test call ---
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words."
)

print(response.text)
