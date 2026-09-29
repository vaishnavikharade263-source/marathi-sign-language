"""
recognize.py
-------------
Real-time recognition: Camera -> MediaPipe landmarks -> trained classifier
-> majority-vote smoothing -> Marathi letter -> (optional) text-to-speech,
matching the final "Marathi Letter + Accuracy" -> "End" stage of your
flowchart. Logs each recognized letter + confidence to CSV.

Run:
    python recognize.py            # on-screen text only
    python recognize.py --speak    # also speaks the recognized letter aloud
"""

import argparse
import os
import sys
import threading
import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp

from config import LETTER_MAP, MODEL_PATH, SMOOTHING_WINDOW
from feature_utils import extract_features
from db import log_recognition
from text_overlay import draw_text

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Phonetic fallback: pyttsx3's Windows SAPI5 driver can't encode Devanagari
# (UnicodeEncodeError), so on Windows we speak a Latin transliteration.
# Extend/adjust to match the letters in your config.LETTER_MAP.
TRANSLITERATION = {
    "अ": "A", "आ": "Aa", "इ": "I", "ई": "Ee", "उ": "U", "ऊ": "Oo", "ऋ": "Ru",
    "ए": "Ay", "ऐ": "Ai", "ओ": "O", "औ": "Au", "अं": "Am", "अः": "Aha",
    "क": "Ka", "ख": "Kha", "ग": "Ga", "घ": "Gha", "ङ": "Nga",
    "च": "Cha", "छ": "Chha", "ज": "Ja", "झ": "Jha", "ञ": "Nya",
    "ट": "Ta", "ठ": "Tha", "ड": "Da", "ढ": "Dha", "ण": "Na",
    "त": "Ta", "थ": "Tha", "द": "Da", "ध": "Dha", "न": "Na",
    "प": "Pa", "फ": "Pha", "ब": "Ba", "भ": "Bha", "म": "Ma",
    "य": "Ya", "र": "Ra", "ल": "La", "व": "Va", "श": "Sha",
    "ष": "Sha", "स": "Sa", "ह": "Ha", "ळ": "La", "क्ष": "Ksha", "ज्ञ": "Dnya",
}

_speech_lock = threading.Lock()


def _speak_blocking(text: str):
    if not _speech_lock.acquire(blocking=False):
        return  # already speaking; skip rather than queue up stale letters
    try:
        import pyttsx3
        spoken = TRANSLITERATION.get(text, text) if sys.platform.startswith("win") else text
        engine = pyttsx3.init()
        engine.say(spoken)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS unavailable: {e}] Recognized letter: {text}")
    finally:
        _speech_lock.release()


def speak(text: str):
    """Non-blocking TTS: runs in a daemon thread so the video loop never freezes."""
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


class SmoothedRecognizer:
    """
    Wraps a trained model with a rolling majority-vote buffer. A single
    frame's prediction can flicker (occlusion, motion blur, borderline
    hand pose); voting over the last SMOOTHING_WINDOW predictions removes
    most of that flicker essentially for free -- each individual prediction
    is sub-millisecond, so buffering a handful of them adds no perceptible
    latency while visibly raising real-world accuracy.
    """

    def __init__(self, model, window: int = SMOOTHING_WINDOW, confidence_threshold: float = 0.55):
        self.model = model
        self.buffer = deque(maxlen=window)
        self.confidence_threshold = confidence_threshold

    def update(self, features):
        proba = self.model.predict_proba([features])[0]
        best_idx = proba.argmax()
        confidence = proba[best_idx]
        label = self.model.classes_[best_idx]

        if confidence >= self.confidence_threshold:
            self.buffer.append(label)
        elif self.buffer:
            # Low-confidence frame: let old votes fade instead of holding a stale letter forever
            self.buffer.popleft()

        if not self.buffer:
            return None, None, confidence

        voted_label, votes = Counter(self.buffer).most_common(1)[0]
        vote_confidence = votes / len(self.buffer)
        return voted_label, LETTER_MAP[voted_label], vote_confidence


def run(speak_output: bool = False, cam_index: int = 0):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    recognizer = SmoothedRecognizer(bundle["model"])
    print(f"Loaded {bundle['model_kind']} model from {MODEL_PATH}")

    cap = cv2.VideoCapture(cam_index)
    last_spoken, last_speak_time = None, 0
    last_logged_label, last_log_time = None, 0.0
    fps_smoothed, prev_frame_time = 0.0, time.time()

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    ) as hands:

        print("Press 'q' to quit.")
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            display_text = "No hand detected"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = extract_features(hand_landmarks.landmark)
                letter_key, marathi_letter, confidence = recognizer.update(features)

                if marathi_letter:
                    display_text = f"{marathi_letter}  ({confidence:.0%} agreement)"
                    now = time.time()
                    # Rate-limit DB writes: on change, or every 1.5s while held
                    if letter_key != last_logged_label or now - last_log_time >= 1.5:
                        log_recognition(letter_key, marathi_letter, confidence)
                        last_logged_label, last_log_time = letter_key, now

                    if speak_output and (marathi_letter != last_spoken or now - last_speak_time > 2):
                        speak(marathi_letter)
                        last_spoken, last_speak_time = marathi_letter, now
                else:
                    display_text = "Uncertain / stabilizing..."
            else:
                recognizer.buffer.clear()
                last_logged_label = None

            # ---- FPS counter (evidence for the "faster" claim in your report) ----
            now = time.time()
            fps = 1.0 / max(now - prev_frame_time, 1e-6)
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps if fps_smoothed else fps
            prev_frame_time = now

            # cv2.putText can't draw Devanagari at all, so the letter/status
            # text goes through draw_text (Pillow-based) instead; the FPS
            # counter is plain ASCII so cv2.putText is fine for it.
            frame = draw_text(frame, display_text, position=(10, 30), font_size=36, color=(0, 255, 0))
            cv2.putText(frame, f"FPS: {fps_smoothed:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Marathi Sign Language Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Marathi sign language recognition.")
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech output")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    run(speak_output=args.speak, cam_index=args.cam)
