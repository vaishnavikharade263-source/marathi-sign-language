"""
preprocessing.py
-----------------
Dataset validation, now operating on the `landmarks` table in
data/marathi_slr.db instead of CSV files. Same six steps as before:

    1. Integrity check   - valid label, correct feature length
    2. Duplicate removal  - drop exact-duplicate feature vectors
    3. Missing value handling - drop rows with NaN/invalid feature values
    4. Quality filtering   - drop low-variance / degenerate landmark rows
    5. Class balancing      - equalize sample counts per letter
    6. Train / Validation / Test split

Instead of writing new CSVs, this UPDATEs each row's `split` column in
place: 'train', 'val', 'test', or 'excluded' (rows that failed
validation -- kept in the table for auditability rather than deleted, so
you can always see exactly what was filtered out and why by querying the
database directly, e.g. `SELECT * FROM landmarks WHERE split = 'excluded'`).

Run:
    python preprocessing.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import LETTER_MAP
from db import get_connection, decode_features
from feature_utils import NUM_FEATURES

if hasattr(sys.stdout, "reconfigure"):  # avoid UnicodeEncodeError on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_landmarks_df() -> pd.DataFrame:
    with get_connection() as conn:
        raw = pd.read_sql_query("SELECT id, label, features, source FROM landmarks", conn)
    raw["feature_array"] = raw["features"].apply(decode_features)
    return raw


def integrity_check(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/6] Integrity check")
    before = len(df)

    valid_label_mask = df["label"].isin(LETTER_MAP.keys())
    correct_length_mask = df["feature_array"].apply(lambda a: len(a) == NUM_FEATURES)
    df = df[valid_label_mask & correct_length_mask]

    print(f"  - Dropped {before - len(df)} rows with invalid labels or malformed feature vectors")
    print(f"  - {len(df)} rows have valid schema and labels")
    return df.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] Removing duplicates")
    before = len(df)
    df = df.drop_duplicates(subset=["features"])  # exact-byte comparison on the raw blob
    print(f"  - Removed {before - len(df)} exact duplicate feature vectors")
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Handling missing values")
    before = len(df)
    valid_mask = df["feature_array"].apply(lambda a: not np.isnan(a).any())
    df = df[valid_mask]
    print(f"  - Dropped {before - len(df)} rows with NaN/invalid feature values")
    return df.reset_index(drop=True)


def quality_filter(df: pd.DataFrame, min_std: float = 1e-4) -> pd.DataFrame:
    print("\n[4/6] Quality filtering (degenerate / low-variance samples)")
    before = len(df)
    stds = df["feature_array"].apply(lambda a: np.std(a))
    df = df[stds > min_std]
    print(f"  - Removed {before - len(df)} low-quality / degenerate samples")
    return df.reset_index(drop=True)


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/6] Balancing classes")
    counts = df["label"].value_counts()
    print("  - Class counts before balancing (top 5 / bottom 5):")
    for label, count in pd.concat([counts.head(5), counts.tail(5)]).items():
        print(f"      {LETTER_MAP.get(label, label)}: {count}")

    target = counts.min()
    parts = [df[df.label == lbl].sample(target, random_state=42) for lbl in counts.index]
    balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  - Balanced to {target} samples per class ({len(counts)} classes -> {len(balanced)} rows)")
    return balanced


def split_dataset(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42):
    print("\n[6/6] Train / Validation / Test split")
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df["label"], random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=seed
    )
    print(f"  - Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


def apply_splits(all_ids: set, train_ids, val_ids, test_ids):
    """Write the split assignment back into the landmarks table."""
    excluded_ids = all_ids - set(train_ids) - set(val_ids) - set(test_ids)
    with get_connection() as conn:
        conn.executemany("UPDATE landmarks SET split = 'train' WHERE id = ?", [(i,) for i in train_ids])
        conn.executemany("UPDATE landmarks SET split = 'val' WHERE id = ?", [(i,) for i in val_ids])
        conn.executemany("UPDATE landmarks SET split = 'test' WHERE id = ?", [(i,) for i in test_ids])
        conn.executemany("UPDATE landmarks SET split = 'excluded' WHERE id = ?", [(i,) for i in excluded_ids])
    print(f"\nWrote split labels back to the database "
          f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}, excluded={len(excluded_ids)})")


def run_pipeline():
    df = load_landmarks_df()
    all_ids = set(df["id"])
    print(f"Loaded {len(df)} rows from data/marathi_slr.db")

    df = integrity_check(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = quality_filter(df)
    df = balance_classes(df)

    train_df, val_df, test_df = split_dataset(df)
    apply_splits(all_ids, train_df["id"], val_df["id"], test_df["id"])


if __name__ == "__main__":
    run_pipeline()
