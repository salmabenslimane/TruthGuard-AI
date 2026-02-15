"""
Configuration file for TruthGuard-AI project
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_CACHE_DIR = MODEL_DIR / "cache"  # HuggingFace cache
LOG_DIR = BASE_DIR / "logs"

# Model configuration
MODEL_NAME = "roberta-base"
NUM_LABELS = 6
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Label mapping for LIAR dataset
LABEL_MAP = {
    0: "Pants on Fire",
    1: "False",
    2: "Barely True",
    3: "Half True",
    4: "Mostly True",
    5: "True"
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# LIAR dataset label mapping (string to int)
LIAR_LABEL_MAPPING = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Training configuration
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "roberta_fact_checker"))

# Processed data files
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "valid_processed.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

# Set HuggingFace cache directory
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, MODEL_CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)