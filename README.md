# 🎙️ Multilingual Voice Classification Dataset (960 Audio Samples across 13 Languages)

A comprehensive, benchmark-compliant dataset designed for training machine learning and deep learning models to perform **AI vs. Human Voice Detection across 13 languages**.

---

## 📌 Benchmark Rules Applied (as per Dataset Guidelines)

| Factor | AI Voices (480 Samples) | Human Voices (480 Samples) |
| :--- | :--- | :--- |
| **Duration** | **3 to 5 seconds per audio** (micro-sample attacks) | **3 to 5 seconds per audio** |
| **Diversity** | Multi-TTS engines across 13 languages (Edge-TTS, Neural Models, Varied Pitch/Rate) | Diverse native speakers across age & gender (Male, Female) |
| **Audio Format** | **16 kHz Mono WAV** raw audio | **16 kHz Mono WAV** clear voice + realistic subtle background noise |

---

## 🔤 Language Code Mapping Reference

The dataset uses standard 2-letter ISO language codes in file naming conventions (e.g., `human_hi_001.wav`, `ai_hi_001.wav` for Hindi):

| Language Name | Language Code | Human Audio Clips | AI Audio Clips | Total Audio Files | Sample Filename Example |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **English** | `en` | 130 | 130 | **260** | `human_en_001.wav` / `ai_en_001.wav` |
| **Hindi** | `hi` | 100 | 100 | **200** | `human_hi_001.wav` / `ai_hi_001.wav` |
| **Marathi** | `mr` | 50 | 50 | **100** | `human_mr_001.wav` / `ai_mr_001.wav` |
| **Gujarati** | `gu` | 20 | 20 | **40** | `human_gu_001.wav` / `ai_gu_001.wav` |
| **Bengali** | `bn` | 20 | 20 | **40** | `human_bn_001.wav` / `ai_bn_001.wav` |
| **Telugu** | `te` | 20 | 20 | **40** | `human_te_001.wav` / `ai_te_001.wav` |
| **Tamil** | `ta` | 20 | 20 | **40** | `human_ta_001.wav` / `ai_ta_001.wav` |
| **Urdu** | `ur` | 20 | 20 | **40** | `human_ur_001.wav` / `ai_ur_001.wav` |
| **Kannada** | `kn` | 20 | 20 | **40** | `human_kn_001.wav` / `ai_kn_001.wav` |
| **Punjabi** | `pa` | 20 | 20 | **40** | `human_pa_001.wav` / `ai_pa_001.wav` |
| **Odia** | `or` | 20 | 20 | **40** | `human_or_001.wav` / `ai_or_001.wav` |
| **Assamese** | `as` | 20 | 20 | **40** | `human_as_001.wav` / `ai_as_001.wav` |
| **Malayalam** | `ml` | 20 | 20 | **40** | `human_ml_001.wav` / `ai_ml_001.wav` |
| **TOTAL** | | **480** | **480** | **960** | **13 Languages** |

---

## 🗂️ Repository Directory Structure

```text
voice-data/
├── voice data/
│   ├── human/           # 480 Human voice 16kHz WAV files (3-5s duration, ambient noise)
│   └── ai/              # 480 Neural AI voice 16kHz WAV files (3-5s duration, raw audio)
├── metadata.csv         # Full 960-sample metadata index (labels, languages, speakers)
├── features.csv         # 38 extracted acoustic features per audio sample
├── build_dataset.py     # Multilingual dataset generator script
├── extract_features.py  # Audio feature extraction pipeline (MFCCs, Spectral, Pitch)
├── train_baseline.py    # Model training script (Random Forest & SVM)
├── predict.py           # Inference script for testing custom audio clips
├── voice_classifier.pkl # Saved trained classifier model & scaler
└── README.md            # Benchmark Guidelines & Documentation
```

---

## 📊 Model Performance

- **Dataset Size**: 960 audio samples (480 Human, 480 AI)
- **Random Forest Cross-Validation Accuracy**: **99.48%**
- **Holdout Test Set Accuracy**: **99.58%**

---

## 🚀 Quick Usage Commands

```bash
# Extract acoustic features
python extract_features.py

# Train baseline classifier
python train_baseline.py

# Predict voice class on any audio clip
python predict.py "voice data/human/human_hi_001.wav"
python predict.py "voice data/ai/ai_hi_001.wav"
```
