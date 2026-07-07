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
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100