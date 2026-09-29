"""
data_collection.py
-------------------
Optional supplementary webcam collection -- your main training data now
comes from build_landmark_dataset.py (your parquet file). Use this script
if you want to add a few of your own live samples on top of that (e.g. to
personalize the model to your own hand, or pad out a letter that MediaPipe
struggled to detect in the dataset images).

Captures webcam frames, detects a hand with MediaPipe, extracts the 78-dim
feature vector (feature_utils.py), and inserts it into the SAME `landmarks`
table in data/marathi_slr.db that build_landmark_dataset.py writes to,
tagged source='webcam' so you can always tell dataset-derived vs.
self-recorded samples apart later.

Run:
    python data_collection.py --letter अ --samples 200
    python data_collection.py --label 0 --samples 200   # equivalent, by index
"""

import argparse
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp

from config import LETTER_MAP, REVERSE_LETTER_MAP
from feature_utils import extract_features
from db import insert_landmark
from text_overlay import draw_text

if hasattr(sys.stdout, "reconfigure"):  # avoid UnicodeEncodeError on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def resolve_label(letter: Optional[str], label: Optional[int]) -> int:
    if label is not None:
        if label not in LETTER_MAP:
            raise ValueError(f"label {label} is not a valid index (0-{len(LETTER_MAP) - 1})")
        return label
    if letter is not None:
        if letter not in REVERSE_LETTER_MAP:
            raise ValueError(f"'{letter}' is not a recognized Marathi letter in LETTER_MAP")
        return REVERSE_LETTER_MAP[letter]
    raise ValueError("Provide either --letter or --label")


def collect(label: int, num_samples: int = 200, cam_index: int = 0):
    marathi_letter = LETTER_MAP[label]

    cap = cv2.VideoCapture(cam_index)
    collected = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        print(f"Collecting samples for '{marathi_letter}' (label {label}). Press 'q' to stop early.")

        while cap.isOpened() and collected < num_samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = extract_features(hand_landmarks.landmark)
                insert_landmark(label, features, source="webcam")
                collected += 1

            frame = draw_text(frame, f"{marathi_letter} | {collected}/{num_samples}",
                              position=(10, 10), font_size=36, color=(0, 255, 0))
            cv2.imshow("Data Collection - Marathi Sign Language", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {collected} samples for '{marathi_letter}'. "
          f"Inserted into data/marathi_slr.db (source='webcam').")
    print("Re-run preprocessing.py so these new rows get validated and assigned a split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect extra MediaPipe hand-landmark samples for a Marathi letter.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--letter", help="Marathi letter, e.g. अ")
    group.add_argument("--label", type=int, help="Integer label index, e.g. 0 for अ")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    resolved_label = resolve_label(args.letter, args.label)
    collect(resolved_label, args.samples, args.cam)
