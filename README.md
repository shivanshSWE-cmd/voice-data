# 🎙️ Massive Multilingual Voice Classification Dataset (960 Audio Samples across 13 Languages)

A comprehensive, ready-to-train dataset designed for training machine learning and deep learning models to perform **AI vs. Human Voice Detection across 13 languages**.

---

## 📁 Multilingual Dataset Breakdown (960 Total Audio Samples)

| Language | Code | AI Voice Samples | Human Voice Samples | Total Samples | Key Models & Speech Sources |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **English** | `en` | 130 | 130 | **260** | US/UK/AU Neural TTS & LibriSpeech |
| **Hindi** | `hi` | 100 | 100 | **200** | Swara & Madhur Neural TTS & OpenSLR 103 |
| **Marathi** | `mr` | 50 | 50 | **100** | Aarohi & Manohar Neural TTS & OpenSLR 64 |
| **Gujarati** | `gu` | 20 | 20 | **40** | Dhwani & Niranjan Neural TTS & OpenSLR 103 |
| **Bengali** | `bn` | 20 | 20 | **40** | Tanishaa & Bashkar Neural TTS & OpenSLR 103 |
| **Telugu** | `te` | 20 | 20 | **40** | Shruti & Mohan Neural TTS & OpenSLR 103 |
| **Tamil** | `ta` | 20 | 20 | **40** | Pallavi & Valluvar Neural TTS & OpenSLR 103 |
| **Urdu** | `ur` | 20 | 20 | **40** | Gul & Salman Neural TTS & OpenSLR Corpus |
| **Kannada** | `kn` | 20 | 20 | **40** | Sapna & Gagan Neural TTS & OpenSLR 103 |
| **Punjabi** | `pa` | 20 | 20 | **40** | Neural TTS & OpenSLR Speech Corpus |
| **Odia** | `or` | 20 | 20 | **40** | Subhasini Neural TTS & OpenSLR 103 |
| **Assamese** | `as` | 20 | 20 | **40** | Neural TTS & OpenSLR Speech Corpus |
| **Malayalam** | `ml` | 20 | 20 | **40** | Sobhana & Midhun Neural TTS & OpenSLR 103 |
| **TOTAL** | | **480** | **480** | **960** | **13 Major Languages** |

---

## 🗂️ Repository Directory Structure

```text
voice-data/
├── voice data/
│   ├── human/           # 480 Human voice .wav audio files across 13 languages
│   └── ai/              # 480 Neural AI voice .wav audio files across 13 languages
├── metadata.csv         # Full 960-sample metadata index (labels, languages, speakers)
├── features.csv         # 38 extracted acoustic features per audio sample
├── build_dataset.py     # Multilingual dataset generator script
├── extract_features.py  # Audio feature extraction pipeline (MFCCs, Spectral, Pitch)
├── train_baseline.py    # Model training script (Random Forest & SVM)
├── predict.py           # Inference script for testing custom audio clips
├── voice_classifier.pkl # Saved trained classifier model & scaler
└── README.md
```

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
