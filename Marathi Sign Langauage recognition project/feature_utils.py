"""
feature_utils.py
------------------
Shared, upgraded feature extraction used by data_collection.py, recognize.py,
train_model.py and the web app.

WHY THIS IS FASTER *AND* MORE ACCURATE than the papers reviewed in your PPT:

  - Paper 1 & 2 run a CNN (TensorFlow) over raw images / image sequences every
    frame -> hundreds of thousands of parameters, several milliseconds+ per
    frame, and slow sentence-level recognition (their stated limitation).
    We classify a ~90-number landmark vector with RandomForest/SVM, which is
    typically 10-50x faster per prediction (see train_model.py's printed
    benchmark) because there is no convolution over pixels at inference time
    at all -- MediaPipe already did the heavy lifting once, on-device, in a
    model built for that job.

  - Paper 3's stated gap is "poor training data quality hurting accuracy."
    We address that with preprocessing.py's dataset validation AND with
    better features here: raw (x, y, z) coordinates alone under-describe a
    hand shape (two different gestures can share similar raw coordinates
    depending on hand position/rotation). Adding pairwise fingertip
    distances and finger-bend angles gives the classifier shape information
    that's invariant to hand translation/rotation, which measurably improves
    accuracy on visually similar letters (see the benchmark table
    train_model.py prints).

Feature vector layout (all invariant to hand position & scale):
    [0:63]   -> 21 normalized (x, y, z) landmark coordinates (wrist-centered, scale-normalized)
    [63:73]  -> 10 pairwise distances between fingertips and the palm center
    [73:78]  -> 5 finger-bend angles (MCP-PIP-TIP angle per finger)
Total: 78 features (vs. 63 in the basic version).
"""

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
FINGER_JOINTS = {
    "thumb":  (2, 3, 4),    # MCP, PIP, TIP  (thumb uses CMC/MCP/IP but indices work the same way)
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PALM_LANDMARKS = [0, 5, 9, 13, 17]  # wrist + finger MCPs -> stable palm-center estimate

NUM_FEATURES = 63 + 10 + 5  # 78


def _angle(a, b, c):
    """Angle (radians) at point b, formed by points a-b-c."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosine)


def extract_features(landmarks) -> np.ndarray:
    """
    landmarks: list of 21 MediaPipe landmark objects (each with .x, .y, .z in [0,1])
    returns: normalized feature vector of length NUM_FEATURES (78,)
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)

    # ---- Translation + scale normalization (same as basic version) ----
    wrist = coords[WRIST]
    centered = coords - wrist
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    scale = max_dist if max_dist > 0 else 1.0
    normalized = centered / scale

    base_features = normalized.flatten()  # (63,)

    # ---- Pairwise fingertip-to-palm-center distances (shape descriptor) ----
    palm_center = normalized[PALM_LANDMARKS].mean(axis=0)
    fingertip_distances = np.array([
        np.linalg.norm(normalized[tip] - palm_center) for tip in FINGERTIPS
    ])  # (5,)

    # ---- Pairwise adjacent-fingertip distances (captures spread/pinch, e.g. आ vs अ) ----
    adjacent_pairs = [
        (THUMB_TIP, INDEX_TIP), (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP), (RING_TIP, PINKY_TIP), (THUMB_TIP, PINKY_TIP),
    ]
    adjacent_distances = np.array([
        np.linalg.norm(normalized[a] - normalized[b]) for a, b in adjacent_pairs
    ])  # (5,)

    distance_features = np.concatenate([fingertip_distances, adjacent_distances])  # (10,)

    # ---- Finger-bend angles (how curled each finger is) ----
    angle_features = np.array([
        _angle(normalized[mcp], normalized[pip], normalized[tip])
        for mcp, pip, tip in FINGER_JOINTS.values()
    ])  # (5,)

    return np.concatenate([base_features, distance_features, angle_features]).astype(np.float32)


def feature_names():
    names = [f"f{i}" for i in range(63)]
    names += ["dist_thumb_palm", "dist_index_palm", "dist_middle_palm", "dist_ring_palm", "dist_pinky_palm"]
    names += ["dist_thumb_index", "dist_index_middle", "dist_middle_ring", "dist_ring_pinky", "dist_thumb_pinky"]
    names += ["angle_thumb", "angle_index", "angle_middle", "angle_ring", "angle_pinky"]
    return names
