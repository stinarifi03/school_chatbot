import os
from pathlib import Path

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Paths - go up one level from src to school-chatbot, then into data
BASE_DIR = Path(__file__).parent.parent  # This goes from src -> school-chatbot
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
FAQS_DIR = DATA_DIR / "faqs"

# Embedding model (fast and accurate)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for the above model

# Chunking settings - IMPROVED
CHUNK_SIZE = 500  # Slightly smaller for more precise chunks
CHUNK_OVERLAP = 100  # Good overlap to maintain context

# Retrieval settings - IMPROVED
TOP_K = 8  # Increased from 3 to get more context
SCORE_THRESHOLD = 0.25  # Lowered from 0.7 to be less strict and find more relevant info

# FAISS index settings
FAISS_INDEX_PATH = PROCESSED_DIR / "epoka_index.faiss"
METADATA_PATH = PROCESSED_DIR / "metadata.pkl"

# Create directories
for directory in [PDF_DIR, PROCESSED_DIR, FAQS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)