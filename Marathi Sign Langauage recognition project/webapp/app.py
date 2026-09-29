"""
app.py
-------
Flask web app for Marathi Sign Language Recognition (Approach 3 from your
PPT: "Camera -> Recognition model -> Marathi text -> Text-to-Speech in
browser").

Design choices made for SPEED:
  - The heavy work (MediaPipe landmark extraction + classifier prediction)
    happens once per frame on the server and is streamed to the browser as
    MJPEG -- no per-frame round trip of raw image bytes to a separate
    inference call.
  - Text-to-speech runs with the browser's built-in Web Speech API
    (`speechSynthesis`), which speaks instantly client-side with zero
    server load or network round-trip, instead of generating and streaming
    an audio file from the server on every recognition (which is what adds
    noticeable lag in a naive Flask+gTTS setup).
  - The frontend polls a tiny JSON endpoint (~10 KB/s) for the current
    letter rather than re-fetching anything heavy.

Run:
    python app.py
Then open http://localhost:5000 in a browser.
"""

import os
import sys
import threading
import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template

# Allow importing the project's shared modules (config, feature_utils, db) from
# the parent directory without needing to install the project as a package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LETTER_MAP, REVERSE_LETTER_MAP, REFERENCE_CHART_SECTIONS, MODEL_PATH, SMOOTHING_WINDOW
from feature_utils import extract_features
from db import log_recognition, recognition_log_counts, recognition_log_total

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------------------
# Global recognition state, updated by the video-streaming generator and
# read by the polling endpoint. A Flask dev server handles this single
# shared camera/model fine for a mini-project demo (one user at a time).
# ---------------------------------------------------------------------------
_state = {"letter": None, "label": None, "confidence": 0.0, "fps": 0.0, "running": False}
_camera_lock = threading.Lock()   # only one stream may own the webcam at a time
_stream_id = 0   # bumped by each new /video_feed so a stale generator can't stop a newer one

_model_bundle = None
if os.path.exists(MODEL_PATH):
    _model_bundle = joblib.load(MODEL_PATH)


def gen_frames():
    """MJPEG generator: capture -> MediaPipe -> classify -> annotate -> yield JPEG."""
    if _model_bundle is None:
        raise RuntimeError("No trained model found. Run train_model.py from the project root first.")

    global _stream_id
    _stream_id += 1
    my_id = _stream_id
    model = _model_bundle["model"]
    # Per-stream buffer: a shared global one could be emptied by another stream
    # between `if buffer:` and `most_common(1)[0]` -> IndexError.
    buffer = deque(maxlen=SMOOTHING_WINDOW)

    # Wait for a previous stream (Stop/refresh) to release the webcam first.
    if not _camera_lock.acquire(timeout=5):
        return
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        for _ in range(10):                      # camera can take a moment to free up on Windows
            if cap.isOpened():
                break
            time.sleep(0.2)
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        if my_id != _stream_id:                  # a newer stream was requested while we waited
            return
        yield from _stream_loop(cap, model, buffer, my_id)
    finally:
        if cap is not None:
            cap.release()      # always free the webcam, even if the client disconnects
        if my_id == _stream_id:   # only reset shared state if no newer stream took over
            _state.update(running=False, letter=None, label=None, confidence=0.0)
        _camera_lock.release()


def _stream_loop(cap, model, buffer, my_id):
    prev_time = time.time()
    _state["running"] = True
    no_hand_frames = 0          # consecutive frames with no hand detected
    last_logged_label = None    # last letter written to the database
    last_log_time = 0.0

    if True:
        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7,
        ) as hands:
            while _state["running"] and my_id == _stream_id:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                display_text = "No hand detected"

                if result.multi_hand_landmarks:
                    no_hand_frames = 0
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    features = extract_features(hand_landmarks.landmark)
                    proba = model.predict_proba([features])[0]
                    best_idx = proba.argmax()
                    confidence = float(proba[best_idx])
                    # Cast to a plain Python int: scikit-learn's model.classes_ is a
                    # numpy array, so indexing it returns a numpy.int32/int64 scalar,
                    # not a native Python int. Flask's jsonify() can't serialize that
                    # numpy type, which is what was crashing /current_letter.
                    label = int(model.classes_[best_idx])

                    if confidence >= 0.55:
                        buffer.append(label)
                    elif buffer:
                        buffer.popleft()   # fade stale votes on low-confidence frames

                    if buffer:
                        voted_label, votes = Counter(buffer).most_common(1)[0]
                        vote_conf = votes / len(buffer)
                        marathi_letter = LETTER_MAP[voted_label]
                        display_text = f"Agreement {vote_conf:.0%}"

                        _state.update(letter=marathi_letter, label=voted_label, confidence=vote_conf)

                        # Log only when the recognized letter changes (or every 1.5s while
                        # it is held) -- writing to SQLite on every frame was wasteful
                        # and slowed the video down.
                        now_t = time.time()
                        if voted_label != last_logged_label or now_t - last_log_time >= 1.5:
                            log_recognition(voted_label, marathi_letter, vote_conf)
                            last_logged_label, last_log_time = voted_label, now_t
                    else:
                        display_text = "Hold steady..."
                        _state.update(letter=None, label=None, confidence=0.0)
                else:
                    # No hand in view: after ~1 second, clear the old result so the
                    # page doesn't keep showing a stale letter as if it were live.
                    no_hand_frames += 1
                    if no_hand_frames >= 3:
                        buffer.clear()
                        last_logged_label = None
                        _state.update(letter=None, label=None, confidence=0.0)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                _state["fps"] = round(0.9 * _state["fps"] + 0.1 * fps, 1) if _state["fps"] else round(fps, 1)
                prev_time = now

                # The recognized Marathi letter is shown as real text in the panel next to
                # the video (browsers render Devanagari fine); the video overlay only shows
                # plain ASCII status, which cv2.putText can draw reliably.
                cv2.putText(frame, display_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {_state['fps']}", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    letters = list(LETTER_MAP.values())
    model_name = _model_bundle["model_kind"] if _model_bundle else "No model trained yet"
    return render_template("index.html", letters=letters, model_name=model_name,
                            model_ready=_model_bundle is not None)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    _state["running"] = False
    return jsonify({"stopped": True})


@app.route("/current_letter")
def current_letter():
    return jsonify(_state)


@app.route("/reference")
def reference():
    sections = []
    for title, letters in REFERENCE_CHART_SECTIONS:
        # Note: key is "entries", not "items" -- "items" would collide with
        # Python dict's built-in .items() method when accessed via Jinja's
        # dot notation in the template (a classic Jinja2 gotcha).
        entries = [{"letter": letter, "label": REVERSE_LETTER_MAP[letter]} for letter in letters]
        sections.append({"title": title, "entries": entries})
    return render_template("reference.html", sections=sections)


@app.route("/logs")
def logs():
    counts = recognition_log_counts()
    total = recognition_log_total()
    return render_template("logs.html", counts=counts, total=total)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
