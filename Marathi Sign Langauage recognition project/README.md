# Marathi Sign Language Recognition — Website (Camera → Gesture → Speech)

Camera → MediaPipe Hand Landmarks → SQLite → ML Classifier → Marathi
Alphabet → Browser Text-to-Speech. Trained from **your own parquet image
dataset** (all 43 Marathi letters), with **every** piece of storage —
training data and live recognition logs — in a single SQLite database.
No CSV files anywhere in this version.

## Your dataset

Your uploaded parquet file (`train-00000-of-00001.parquet`) contains:
- **50,099 images**, 128×128 JPEGs, one hand gesture per image
- **All 43 letters** of the Marathi alphabet (अ through ह, including
  conjuncts क्ष and ज्ञ) — see the full index-to-letter mapping in
  `config.py`'s `LETTER_MAP`
- An integer `label` column (0–42) matching that mapping exactly, and an
  `image` column with `{bytes, path}` — the standard HuggingFace
  `imagefolder`-export format

## How the pipeline changed for this

| Before | Now |
|---|---|
| 3 letters, synthetic/webcam-only data | **All 43 letters**, sourced from your real parquet dataset |
| `landmarks_raw.csv`, `train/val/test.csv` | **`data/marathi_slr.db`** → `landmarks` table |
| `recognition_logs.csv` | **same `marathi_slr.db`** → `recognition_logs` table |
| Webcam-only data collection | `build_landmark_dataset.py` extracts landmarks from your parquet's images; `data_collection.py` still works too, for adding extra live samples on top |

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Put your parquet file in place

```bash
cp /path/to/train-00000-of-00001.parquet data/
```
(or pass `--parquet /full/path/...` to the next command instead of moving the file)

## 3. Extract landmarks from your dataset into SQLite

```bash
python build_landmark_dataset.py --max-per-class 150
```
This decodes each image, runs MediaPipe Hands over it (static-image mode,
since these are individual photos, not video frames), extracts the 78-dim
feature vector (see `feature_utils.py`), and inserts
`(label, features, source='dataset')` rows into the `landmarks` table in
`data/marathi_slr.db`.

- `--max-per-class 150` (default) caps how many images per letter get
  processed — good for a fast first run (~6,450 images instead of 50,099).
- For your final submission, run the full dataset: `--max-per-class 0`.
  With 50k images this will take a while (MediaPipe processes roughly
  20-60 images/sec on a typical laptop CPU depending on hardware) — the
  script prints progress every 500 images so you can track it.
- Not every image will have a detectable hand (cropping, blur, lighting);
  those are skipped and counted at the end. This is normal.

## 4. Validate & split (same 6-step process, now against SQLite)

```bash
python preprocessing.py
```
Integrity check → duplicate removal → missing-value handling → quality
filtering → class balancing → train/val/test split. Instead of writing new
CSVs, this **updates the `split` column** on each row directly in the
database (`'train'`, `'val'`, `'test'`, or `'excluded'` for anything
filtered out — kept, not deleted, so you can always audit exactly what got
filtered and why with a plain SQL query).

## 5. Train & benchmark

```bash
python train_model.py
```
Same as before: benchmarks RandomForest, SVM, and a small MLP on accuracy
*and* inference speed, picks the best trade-off, saves it to
`models/gesture_classifier.pkl`, and writes full numbers to
`models/benchmark_report.json`.

## 6. Run the website

```bash
cd webapp
python app.py
```
Open **http://localhost:5000**, click **Start camera**. Recognized letter,
confidence, and FPS update live; your browser speaks each new letter aloud
(Web Speech API, `lang="mr-IN"`). Visit **/logs** for a recognition-frequency
breakdown, now pulled straight from the `recognition_logs` SQLite table.

## (Optional) Add your own live samples

```bash
python data_collection.py --letter अ --samples 100
```
Inserts into the same `landmarks` table, tagged `source='webcam'` so you
can always tell dataset-derived vs. self-recorded samples apart later
(e.g. `SELECT source, COUNT(*) FROM landmarks GROUP BY source`). Re-run
`preprocessing.py` afterward so the new rows get validated and assigned a
split.

## Inspecting the database

`data/marathi_slr.db` is a normal SQLite file — open it with
[DB Browser for SQLite](https://sqlitebrowser.org/) or any SQLite CLI to
inspect it directly, which is genuinely useful for your report/demo:

```sql
SELECT split, COUNT(*) FROM landmarks GROUP BY split;
SELECT label, COUNT(*) FROM landmarks WHERE split='excluded' GROUP BY label;
SELECT marathi_letter, COUNT(*) FROM recognition_logs GROUP BY marathi_letter ORDER BY 2 DESC;
```

## Project structure

```
config.py                    - 43-letter mapping, paths, constants
db.py                          - SQLite schema + helpers (replaces all CSV files)
feature_utils.py                 - 78-dim feature extraction (coords + distances + angles)
build_landmark_dataset.py          - parquet images -> MediaPipe -> SQLite (main data source)
data_collection.py                   - optional: webcam -> SQLite (source='webcam')
preprocessing.py                       - dataset validation, updates `split` column in SQLite
train_model.py                           - trains + benchmarks RF/SVM/MLP, saves best model
recognize.py                               - desktop real-time inference + TTS, logs to SQLite
webapp/
  app.py                                       - Flask server: MJPEG stream, JSON letter endpoint, logs page
  templates/index.html                           - live camera + recognized letter + browser TTS
  templates/logs.html                              - recognition-frequency breakdown from SQLite
  static/style.css                                   - styling
data/marathi_slr.db                                    - created on first run (landmarks + recognition_logs)
models/                                                  - saved classifier (.pkl) + benchmark_report.json
```

## Notes for your report

- With 43 letters instead of 3, expect real-world accuracy to be lower
  than a 3-class toy example — that's expected and worth stating honestly.
  Some letters look very similar in hand-sign form; your confusion matrix
  (printed by `train_model.py`) will show exactly which pairs the model
  confuses, which is good material for a "limitations and future work"
  section.
- If accuracy on the full 43-class set doesn't clear the 80% threshold
  immediately, the flowchart's own answer is the right one: collect more
  samples (raise `--max-per-class`, or add `data_collection.py` samples
  for the weakest letters) and retrain — the script tells you exactly
  this when the accuracy check fails.
- `source` column on every `landmarks` row (`'dataset'` vs `'webcam'`) is
  a nice thing to mention in your methodology section: it shows exactly
  how much of your training data came from the provided dataset vs. your
  own collection effort.
