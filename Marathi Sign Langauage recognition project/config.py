"""
config.py
---------
Central configuration for the Marathi Sign Language Recognition project.

LETTER_MAP below is the exact 43-class label set extracted from your
uploaded parquet file's embedded HuggingFace `ClassLabel` metadata (key
"huggingface" in the parquet footer) -- these integer indices are exactly
what appears in the parquet's `label` column, so they line up automatically
with `build_landmark_dataset.py`.
"""

import os

# ---------------------------------------------------------------------------
# Full Marathi alphabet, index -> letter, exactly matching your parquet
# file's label encoding (0-42). This replaces the old 3-letter demo set.
# ---------------------------------------------------------------------------
LETTER_MAP = {
    0: "अ", 1: "आ", 2: "इ", 3: "ई", 4: "उ", 5: "ऊ", 6: "ए", 7: "ऐ", 8: "ओ", 9: "औ",
    10: "क", 11: "क्ष", 12: "ख", 13: "ग", 14: "घ", 15: "च", 16: "छ", 17: "ज", 18: "ज्ञ",
    19: "झ", 20: "ट", 21: "ठ", 22: "ड", 23: "ढ", 24: "ण", 25: "त", 26: "थ", 27: "द",
    28: "ध", 29: "न", 30: "प", 31: "फ", 32: "ब", 33: "भ", 34: "म", 35: "य", 36: "र",
    37: "ल", 38: "ळ", 39: "व", 40: "श", 41: "स", 42: "ह",
}

# Reverse map: Marathi letter -> integer label (handy for CLI args, lookups)
REVERSE_LETTER_MAP = {v: k for k, v in LETTER_MAP.items()}

# Traditional Marathi varnamala teaching order (vowels, then consonants,
# then conjuncts) -- used for the reference chart page, since the raw
# 0-42 dataset label order above interleaves the two conjuncts oddly
# (क्ष right after क, ज्ञ right after ज) rather than grouping them.
REFERENCE_CHART_SECTIONS = [
    ("Vowels (स्वर)", ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ"]),
    ("Consonants (व्यंजन)", ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ", "ड", "ढ", "ण",
                              "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल",
                              "व", "श", "स", "ह", "ळ"]),
    ("Conjuncts (जोडाक्षरे)", ["क्ष", "ज्ञ"]),
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Point this at wherever you keep the parquet file (default: data/ folder)
DEFAULT_PARQUET_PATH = os.path.join(DATA_DIR, "train-00000-of-00001.parquet")

# Single SQLite database for EVERYTHING -- landmarks/features (built from the
# parquet + optional webcam collection) and live recognition logs. This
# replaces every CSV file used in the earlier version of this project.
DB_PATH = os.path.join(DATA_DIR, "marathi_slr.db")

MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe / model parameters
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21

from feature_utils import NUM_FEATURES  # 78: 63 coords + 10 distances + 5 angles

ACCURACY_THRESHOLD = 0.80        # matches your flowchart's "accuracy >= 80%" decision node
SMOOTHING_WINDOW = 5              # rolling majority-vote window for recognize.py / webapp
