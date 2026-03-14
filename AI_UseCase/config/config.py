import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR.parent / "data").resolve()
INDEX_DIR = (DATA_DIR / "index").resolve()

load_dotenv(BASE_DIR / ".env")

# API keys (set in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Model defaults
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# RAG defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Retrieval thresholds
MIN_CHUNKS = int(os.getenv("MIN_CHUNKS", "2"))
SIMILARITY_K_MULTIPLIER = int(os.getenv("SIMILARITY_K_MULTIPLIER", "5"))
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "6"))

# OCR (optional)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# Chat history persistence
CHAT_HISTORY_PATH = Path(os.getenv("CHAT_HISTORY_PATH", str(DATA_DIR / "chat_history.json")))
