from pathlib import Path
import os
from typing import Optional
from huggingface_hub import hf_hub_download

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
LOGS_DIR = BASE_DIR / "logs"

DATASET_PATH = DATA_DIR / "dataset_clean.csv"            
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
EMBEDDINGS_NPY_PATH = EMBEDDINGS_DIR / "embeddings.npy"
QUESTIONS_META_PATH = EMBEDDINGS_DIR / "questions_meta.json"  

# ---------- Model / Embedding settings ----------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# batch size when generating embeddings (tune by memory)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

FAISS_INDEX_FACTORY = os.getenv("FAISS_INDEX_FACTORY", "IndexFlatIP")

# ---------- Retrieval ----------
TOP_K = int(os.getenv("TOP_K", "5"))           
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.0"))  

# ---------- CSV columns (change if your CSV different) ----------
COL_ID = os.getenv("COL_ID", "id")
COL_QUESTION = os.getenv("COL_QUESTION", "question")
COL_ANSWER = os.getenv("COL_ANSWER", "answer")

# ---------- API / runtime ----------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() in ("1", "true", "yes")

# ---------- Device detection ----------
def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return os.getenv("DEVICE", "cpu")

DEVICE = get_device()

# ---------- Utilities ----------
def ensure_dirs() -> None:
    """Create essential directories if missing."""
    for d in (DATA_DIR, EMBEDDINGS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        
