"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

# Tránh crash 0xC0000005 (access violation) do xung đột OpenMP giữa
# torch / numpy(MKL) / onnxruntime trên Windows khi load model embedding.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# In UTF-8 ra console Windows (cp1252 không in được ký tự ✓ tiếng Việt).
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

load_dotenv()

# --- HuggingFace cache ---
HF_HOME = os.getenv("HF_HOME", "")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

LLM_API_KEY = OPENAI_API_KEY or GEMINI_API_KEY
_default_base_url = (
    "https://api.openai.com/v1" if OPENAI_API_KEY else "https://ai-gateway.antco.ai/v1"
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _default_base_url)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = "gpt-4o-mini"

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
