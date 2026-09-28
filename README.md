# marathi-sign-language
Marathi Sign Language Recognition
📖 Overview
This project aims to build a Python-based system for recognizing Marathi Sign Language gestures and converting them into text and speech. It addresses communication barriers faced by hearing‑impaired individuals, especially in regional contexts where standardized tools are lacking.

Our solution combines MediaPipe hand landmarks, machine learning classifiers, and text-to-speech to deliver a lightweight, real‑time recognition pipeline.

🎯 Problem Statement
Hearing‑impaired individuals face severe communication barriers because:

Sign language is not widely understood.

Human interpreters are often unavailable or compromise privacy.

Regional languages like Marathi lack standardized translation tools compared to ASL.

This project bridges the gap by providing a real‑time, automated recognition system for Marathi alphabets.

📚 Literature Survey
We reviewed three recent papers (2023–2025):

Marathi sign language recognition using Canny’s edge detection – slow conversion speed.

Real‑time Marathi sign language translation (CNN + LSTM) – limited dataset (15 signs only).

Indian sign language recognition with Google Search API – low accuracy due to poor training data.

Gap Identified: None of the existing solutions provide a complete, accurate, and scalable recognition system for all Marathi letters.

🎯 Objectives
To capture and preprocess Marathi sign language gestures using MediaPipe hand landmarks.

To classify and convert recognized gestures into Marathi alphabets and speech using TensorFlow/Keras + Text‑to‑Speech.

💡 Proposed Solution
Best approach:
Camera → MediaPipe landmarks → ML classifier → Marathi alphabet → Speech

✅ Faster and lightweight for real‑time detection.

✅ Balances simplicity, speed, and feasibility for a mini project.

❌ Sensitive to hand visibility, requires landmark‑based training data.

🚀 Innovativeness
First to combine MediaPipe + ML + Pandas + Text‑to‑Speech in one pipeline.

Focused specifically on Marathi alphabets, unlike prior work.

Lightweight, real‑time, and practical for inclusive classrooms.

🛠️ Technology Stack
Language: Python 3.x

Libraries: OpenCV, MediaPipe, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn

UI: Tkinter / Flask

Storage: CSV / SQLite

Speech: pyttsx3 / gTTS

Tools: VS Code, GitHub

📂 Module Breakdown
Data Input – Camera captures gestures.

Data Storage – Save recognized letters + accuracy logs.

Data Processing – MediaPipe + ML classifier.

Output – Display text, generate speech, accuracy graphs.

User Interface – Tkinter/Flask app with start/stop buttons.

Testing & Deployment – Accuracy validation, confusion matrices, final submission.

📅 Timeline (15 Weeks)
Weeks 1–2: Problem & survey

Week 3: Planning & design

Weeks 6–10: Module coding & integration

Weeks 11–12: Testing & fixes

Weeks 13–15: Documentation & final submission

👩‍💻 Team
Divya Biswas (Roll No. 125A3013)

Vaishnavi Kharade (Roll No. 125A3044)

Nandana Nair (Roll No. 125A3064)
Supervisor: Dr. Sulochana Sagar Madachane

📖 References
Mali, Y.K. (2025). Marathi sign language recognition methodology using Canny’s edge detection.

Musale, S. et al. (2023). Indian sign language recognition and search results.

Bhinganiy, S. et al. (2025). Real‑Time Marathi Sign Language Translation.

Gaur, V. et al. (2022). Conversion of Sign Language into Devanagari Text Using CNN.

Dahibavkar, S. et al. (2020). Marathi Sign Language Recognition.
