import os
from dotenv import load_dotenv

# --- Find and load .env from project root ---
# Get the absolute path to the project root (two levels up)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load .env from root
load_dotenv(ENV_PATH)

# --- Read environment variables ---
API_KEY = os.getenv("CHAT_API_KEY")
BASE_URL = os.getenv("CHAT_BASE_URL", "https://chat-ai.academiccloud.de/v1")
MODEL = os.getenv("CHAT_MODEL", "meta-llama-3.1-8b-instruct")
RAG_MODEL = os.getenv("CHAT_RAG_MODEL", "meta-llama-3.1-8b-rag")

# --- Safety check ---
if not API_KEY:
    raise ValueError(f"CHAT_API_KEY not found in {ENV_PATH}")


if __name__ == "__main__":
    print("API_KEY:", API_KEY[:6] + "...")
    print("BASE_URL:", BASE_URL)
    print("MODEL:", MODEL)
