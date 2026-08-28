# 🎙️ Massive Multilingual Voice Classification Dataset (2,900 Samples across 13 Languages)

A production-grade dataset for training machine learning and deep learning models to perform **AI vs. Human Voice Detection across 13 major languages**.

---

## 📊 Dataset Distribution (2,900 Audio Samples)

| Language Name | Language Code | Human Audio Clips | AI Audio Clips | Total Audio Files | Acoustic Augmentation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **English** | `en` | 300 | 300 | **600** | Multi-speaker Neural TTS & LibriSpeech |
| **Hindi** | `hi` | 300 | 300 | **600** | Swara/Madhur TTS & OpenSLR 103 |
| **Marathi** | `mr` | 150 | 150 | **300** | Aarohi/Manohar TTS & OpenSLR 64 |
| **Bengali** | `bn` | 100 | 100 | **200** | Tanishaa/Bashkar TTS & OpenSLR 53 |
| **Telugu** | `te` | 100 | 100 | **200** | Shruti/Mohan TTS & OpenSLR 66 |
| **Tamil** | `ta` | 100 | 100 | **200** | Pallavi/Valluvar TTS & OpenSLR 65 |
| **Gujarati** | `gu` | 75 | 75 | **150** | Dhwani/Niranjan TTS & OpenSLR 78 |
| **Kannada** | `kn` | 75 | 75 | **150** | Sapna/Gagan TTS & OpenSLR 79 |
| **Malayalam** | `ml` | 75 | 75 | **150** | Sobhana/Midhun TTS & OpenSLR 63 |
| **Punjabi** | `pa` | 50 | 50 | **100** | Neural TTS & Indic Speech Corpora |
| **Urdu** | `ur` | 50 | 50 | **100** | Gul/Salman TTS & OpenSLR Corpus |
| **Odia** | `or` | 50 | 50 | **100** | Subhasini TTS & OpenSLR 103 |
| **Assamese** | `as` | 25 | 25 | **50** | Neural TTS & Indic Speech Corpora |
| **TOTAL** | | **1,450** | **1,450** | **2,900** | **13 Major Languages** |

---

## 📌 Benchmark & Quality Standards

- **Duration**: Every clip strictly formatted to **4.0 seconds** (within 3 to 5-second micro-sample attack rule).
- **Sampling Rate**: **16 kHz Mono WAV** raw audio.
- **Real-World Augmentations**:
  - **Human Voices**: 40% of samples augmented with ambient background room noise, office hum, and mic noise (SNR 20–30 dB).
  - **AI Voices**: 40% of samples augmented with minor voice codec compression simulation (WhatsApp/Telegram voice note simulation).

---

## 📈 Model Performance (Ensemble Classifier)

- **Dataset Size**: 2,900 audio samples (1,450 Human, 1,450 AI)
- **Stratified 10-Fold Cross-Validation Accuracy**: **99.69%** (+/- 0.93%)
- **Holdout Test Accuracy (580 samples / 20%)**: **100.00%**
- **Top Discriminative Acoustic Features**: `rms_mean`, `mfcc_12_mean`, `spec_cent_std`, `mfcc_20_std`, `mfcc_16_std`, `spec_roll_std`.

---

## 🗂️ Directory Layout

```text
voice-data/
├── voice data/
│   ├── human/           # 1,450 Human voice 16kHz WAV files (4.0s duration, ambient noise)
│   └── ai/              # 1,450 Neural AI voice 16kHz WAV files (4.0s duration, codec simulated)
├── metadata.csv         # Full 2,900-sample metadata index (labels, languages, speakers)
├── features.csv         # 52 extracted acoustic features per audio sample
├── build_3000_dataset.py# Multilingual dataset generator script
├── extract_features.py  # Fast 52-feature acoustic extraction pipeline
├── train_baseline.py    # Ensemble classifier training script (Random Forest + ExtraTrees + GradientBoosting)
├── predict.py           # Inference script for testing custom audio clips
├── voice_classifier.pkl # Saved trained ensemble classifier model & scaler
└── README.md            # Documentation & Benchmark Rules
```

---

## 🚀 Quick Usage Commands

```bash
# Extract 52 acoustic features
python extract_features.py

# Train ensemble classifier
python train_baseline.py

# Predict voice class on any audio clip
python predict.py "voice data/human/human_hi_001.wav"
python predict.py "voice data/ai/ai_hi_001.wav"
```
