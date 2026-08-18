# 🎙️ Multilingual Voice Classification Dataset (60 Audio Samples: English & Hindi)

A complete, ready-to-train dataset designed for training machine learning and deep learning models to perform **AI vs Human Voice Detection across English and Hindi languages**.

---

## 📁 Dataset Overview

- **Total Audio Samples**: **60 standardized `.wav` files** (16kHz, mono, normalized volume).
- **Human Voice Samples (30 total)**:
  - **20 English Human Voices**: `human_01.wav` to `human_20.wav` (Sourced from LibriSpeech corpus across male/female speakers).
  - **10 Hindi Human Voices**: `human_hindi_01.wav` to `human_hindi_10.wav` (Sourced from OpenSLR 103 Hindi speech corpus).
- **AI Voice Samples (30 total)**:
  - **20 English AI Voices**: `ai_01.wav` to `ai_20.wav` (Generated with 20 distinct Edge Neural TTS models across US, UK, AU, IN, CA, IE accents).
  - **10 Hindi AI Voices**: `ai_hindi_01.wav` to `ai_hindi_10.wav` (Generated with `hi-IN-SwaraNeural` and `hi-IN-MadhurNeural` with varied pitch, rate, and Hindi text).
- **Metadata Index**: [`metadata.csv`](metadata.csv) maps all 60 files to label (`human` vs `ai`), language (`English` vs `Hindi`), speaker ID, gender, accent, duration, and transcript.

---

## 🗂️ Directory Layout

```text
voice_dataset/
├── dataset/
│   ├── human/           # 30 Human voice .wav files (20 English + 10 Hindi)
│   └── ai/              # 30 Neural AI voice .wav files (20 English + 10 Hindi)
├── metadata.csv         # Full 60-sample metadata index
├── features.csv         # 38 extracted acoustic features per audio sample
├── build_dataset.py     # Generator script for dataset creation/expansion
├── extract_features.py  # Feature extraction script (MFCCs, Spectral, Pitch)
├── train_baseline.py    # Baseline model training script
├── predict.py           # Inference script for testing new audio
├── voice_classifier.pkl # Saved trained classifier model & scaler
└── README.md
```

---

## 📈 Model Performance on Multilingual Dataset

- **Stratified 5-Fold Cross-Validation Accuracy**: **93.33%**
- **Test Set Accuracy (Holdout)**: **93.33%**
- **Hindi Human Voice Prediction Accuracy**: **97.00% Confidence**
- **Hindi AI Voice Prediction Accuracy**: **67.00% Confidence**
