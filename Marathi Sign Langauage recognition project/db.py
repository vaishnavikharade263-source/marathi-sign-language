"""
db.py
------
SQLite database layer for the whole project. This replaces every CSV file
used in the earlier version:

    landmarks_raw.csv, landmarks_clean.csv, train/val/test.csv  -> `landmarks` table
    recognition_logs.csv                                          -> `recognition_logs` table

Two tables, one file (`data/marathi_slr.db`):

  landmarks
  -----------------------------------------------------------------
  id           INTEGER PRIMARY KEY
  label        INTEGER   -- 0-42, matches config.LETTER_MAP
  features     BLOB      -- 78 float32s, packed with numpy.tobytes()
  source       TEXT      -- 'dataset' (from your parquet) or 'webcam' (data_collection.py)
  split        TEXT      -- NULL until preprocessing.py runs, then 'train'/'val'/'test'/'excluded'
  created_at   TEXT

  recognition_logs
  -----------------------------------------------------------------
  id            INTEGER PRIMARY KEY
  timestamp      TEXT
  label          INTEGER
  marathi_letter TEXT
  confidence     REAL

Features are stored as a BLOB (raw float32 bytes) rather than 78 separate
columns -- much easier to add/change features later without a schema
migration, and numpy round-trips it in one line (see encode_features /
decode_features below).
"""

import sqlite3
import numpy as np
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS landmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label INTEGER NOT NULL,
    features BLOB NOT NULL,
    source TEXT NOT NULL DEFAULT 'dataset',
    split TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_landmarks_split ON landmarks(split);
CREATE INDEX IF NOT EXISTS idx_landmarks_label ON landmarks(label);

CREATE TABLE IF NOT EXISTS recognition_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    label INTEGER,
    marathi_letter TEXT,
    confidence REAL
);
"""


@contextmanager
def get_connection():
    # timeout: wait up to 5s for a lock instead of raising "database is locked"
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        # WAL lets readers (/logs page) and the writer (live recognition) work
        # at the same time. The mode is persistent in the DB file; re-issuing it
        # is a cheap no-op once set.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")  # safe with WAL, far fewer fsyncs
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def encode_features(feature_vector: np.ndarray) -> bytes:
    return np.asarray(feature_vector, dtype=np.float32).tobytes()


def decode_features(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def insert_landmark(label: int, feature_vector: np.ndarray, source: str = "dataset"):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO landmarks (label, features, source) VALUES (?, ?, ?)",
            (int(label), encode_features(feature_vector), source),
        )


def insert_landmarks_bulk(rows):
    """rows: iterable of (label:int, feature_vector:np.ndarray, source:str)"""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO landmarks (label, features, source) VALUES (?, ?, ?)",
            [(int(label), encode_features(fv), source) for label, fv, source in rows],
        )


def count_landmarks():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0]


def log_recognition(label: int, marathi_letter: str, confidence: float):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO recognition_logs (label, marathi_letter, confidence) VALUES (?, ?, ?)",
            (int(label), marathi_letter, float(confidence)),
        )


def recognition_log_counts():
    """Returns {marathi_letter: count} for the /logs page, most-recognized first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT marathi_letter, COUNT(*) as n FROM recognition_logs "
            "GROUP BY marathi_letter ORDER BY n DESC"
        ).fetchall()
    return {letter: n for letter, n in rows}


def recognition_log_total():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM recognition_logs").fetchone()[0]


init_db()  # safe to call repeatedly -- CREATE TABLE IF NOT EXISTS
