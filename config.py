import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVALUATION_DIR = DATA_DIR / "evaluation"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEVICE = "cpu"
SPACY_MODEL = "en_core_web_sm"
LLM_MODEL = "openai/gpt-oss-20b"  # Groq's replacement for llama-3.1-8b-instant
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

GROQ_PROMPT_PRICE_PER_1M = 0.075    # $0.075 per 1M input tokens
GROQ_COMPLETION_PRICE_PER_1M = 0.30  # $0.30 per 1M output tokens
