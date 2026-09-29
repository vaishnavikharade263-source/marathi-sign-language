"""
build_landmark_dataset.py
---------------------------
Builds the ML-ready training dataset FROM YOUR OWN parquet file (images +
integer labels, in the standard HuggingFace `imagefolder`-style layout:
an `image` column with {bytes, path}, and a `label` column with an int
matching config.LETTER_MAP).

For every image: decode the JPEG bytes -> run MediaPipe Hands (in static
image mode, since each row is an independent photo, not a video frame) ->
extract the 78-dim feature vector (feature_utils.py) -> insert
(label, features, source='dataset') into the `landmarks` table in
data/marathi_slr.db. This fully replaces the CSV-based
data_collection.py / generate_sample_data.py flow as the source of truth
for training data: your real images, not synthetic samples.

Some images won't have a detectable hand (cropping, blur, lighting) --
these are skipped and counted, not inserted. That's expected and is
exactly what preprocessing.py's later "quality filtering" step also guards
against, just applied earlier here at the image stage.

Run:
    python build_landmark_dataset.py --parquet data/train-00000-of-00001.parquet
    python build_landmark_dataset.py --max-per-class 150   # quick test run
    python build_landmark_dataset.py --max-per-class 0     # 0 = no limit, use everything
"""

import argparse
import io
import sys
import time
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

from config import LETTER_MAP, DEFAULT_PARQUET_PATH
from feature_utils import extract_features
from db import insert_landmarks_bulk, count_landmarks

if hasattr(sys.stdout, "reconfigure"):  # avoid UnicodeEncodeError on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

mp_hands = mp.solutions.hands


def decode_image(image_cell) -> np.ndarray:
    """
    image_cell: the value of the 'image' column for one row, as pandas
    loads it from the parquet -- a dict-like {'bytes': ..., 'path': ...}
    for a HuggingFace Image() feature column. Returns an RGB numpy array.
    """
    raw_bytes = image_cell["bytes"] if isinstance(image_cell, dict) else image_cell.get("bytes")
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.array(pil_img)


def build(parquet_path: str, max_per_class: int, batch_size: int = 200, min_detection_confidence: float = 0.5,
          flip: str = "no", reset: bool = False):
    if reset:
        from db import get_connection
        with get_connection() as conn:
            n = conn.execute("DELETE FROM landmarks WHERE source = 'dataset'").rowcount
        print(f"Reset: removed {n} existing dataset rows (webcam rows kept)")
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)  # requires pyarrow: pip install pyarrow
    print(f"Loaded {len(df)} rows, {df['label'].nunique()} unique labels")

    if max_per_class and max_per_class > 0:
        # Built via plain per-group .head() + concat rather than
        # groupby(...).apply(...): newer pandas versions (2.2+) can silently
        # drop the grouping column ("label") from the result of a
        # groupby().apply() call, which breaks the rest of this script.
        # This pattern avoids that version-dependent behavior entirely.
        df = pd.concat(
            [group.head(max_per_class) for _, group in df.groupby("label")],
            ignore_index=True,
        )
        print(f"Capped to {max_per_class} per class -> {len(df)} rows")

    per_class_seen = defaultdict(int)
    per_class_inserted = defaultdict(int)
    skipped_no_hand = 0
    buffer = []
    start = time.time()

    with mp_hands.Hands(
        static_image_mode=True,       # each row is an independent photo, not a video stream
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        # Iterating via zip(df["label"], df["image"]) rather than df.itertuples()
        # deliberately: itertuples() only exposes columns as row.<name> attributes
        # when every column name passes pandas' internal identifier checks, and
        # silently falls back to positional access otherwise depending on the
        # pandas/pyarrow version -- zip() sidesteps that version-dependent gotcha.
        for i, (label, image_cell) in enumerate(zip(df["label"], df["image"]), start=1):
            label = int(label)
            per_class_seen[label] += 1

            try:
                rgb = decode_image(image_cell)
            except Exception as e:
                print(f"  [skip] row {i}: failed to decode image ({e})")
                continue

            # Live inference uses a mirrored (cv2.flip) webcam view, so the dataset
            # images may need mirroring too: "yes" = flipped only, "both" = original + flipped.
            variants = {"no": [rgb], "yes": [np.ascontiguousarray(rgb[:, ::-1])],
                        "both": [rgb, np.ascontiguousarray(rgb[:, ::-1])]}[flip]
            found_any = False
            for variant in variants:
                result = hands.process(variant)
                if not result.multi_hand_landmarks:
                    continue
                found_any = True
                features = extract_features(result.multi_hand_landmarks[0].landmark)
                buffer.append((label, features, "dataset"))
                per_class_inserted[label] += 1
            if not found_any:
                skipped_no_hand += 1
                continue

            if len(buffer) >= batch_size:
                insert_landmarks_bulk(buffer)
                buffer.clear()

            if i % 500 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                print(f"  ...{i}/{len(df)} images processed "
                      f"({rate:.1f} img/s, {skipped_no_hand} skipped so far)")

    if buffer:
        insert_landmarks_bulk(buffer)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Total rows now in landmarks table: {count_landmarks()}")
    print(f"Skipped (no hand detected): {skipped_no_hand}")
    print("\nPer-class counts (seen -> inserted):")
    for label in sorted(per_class_seen):
        letter = LETTER_MAP.get(label, f"label {label}")
        print(f"  {letter:>3s}  seen={per_class_seen[label]:4d}  inserted={per_class_inserted[label]:4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from a parquet image dataset into SQLite.")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET_PATH, help="Path to the parquet file")
    parser.add_argument("--max-per-class", type=int, default=150,
                         help="Cap images processed per class (0 = use all rows; full run is slower)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--flip", choices=["no", "yes", "both"], default="no",
                         help="Mirror dataset images to match the flipped live webcam view "
                              "('both' = keep original AND mirrored copies)")
    parser.add_argument("--reset", action="store_true",
                         help="Delete existing source='dataset' rows first (avoids duplicates on rebuild)")
    args = parser.parse_args()

    build(args.parquet, args.max_per_class, min_detection_confidence=args.min_detection_confidence,
          flip=args.flip, reset=args.reset)
