# 🎙️ Multilingual YouTube Human & Neural AI Voice Dataset (23,500 Samples)

A production-grade dataset for training machine learning and deep learning models to perform **AI vs. Real-World Human Voice Detection** across **13 major languages**, with human audio clips extracted from **authentic YouTube speech streams** (podcasts, speeches, interviews, broadcasts).

---

## 📊 Dataset Distribution Across 13 Languages (23,500 Audio Samples)

| Language Name | Language Code | YouTube Human Clips | Neural AI Clips | Total Audio Files | Source Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **English** | `en` | 2,000 | 2,000 | **4,000** | YouTube Speech Streams & Multi-Speaker Neural TTS |
| **Hindi** | `hi` | 2,000 | 2,000 | **4,000** | YouTube Hindi Podcasts & Swara/Madhur Neural TTS |
| **Marathi** | `mr` | 1,000 | 1,000 | **2,000** | YouTube Marathi Broadcasts & Manohar Neural TTS |
| **Gujarati** | `gu` | 1,000 | 1,000 | **2,000** | YouTube Gujarati Talks & Niranjan Neural TTS |
| **Bengali** | `bn` | 1,000 | 1,000 | **2,000** | YouTube Bengali Speeches & Bashkar Neural TTS |
| **Telugu** | `te` | 1,000 | 1,000 | **2,000** | YouTube Telugu Talks & Mohan Neural TTS |
| **Tamil** | `ta` | 1,000 | 1,000 | **2,000** | YouTube Tamil Discussions & Valluvar Neural TTS |
| **Urdu** | `ur` | 500 | 500 | **1,000** | YouTube Urdu Discussions & Salman Neural TTS |
| **Kannada** | `kn` | 500 | 500 | **1,000** | YouTube Kannada Broadcasts & Gagan Neural TTS |
| **Malayalam** | `ml` | 500 | 500 | **1,000** | YouTube Malayalam Podcasts & Midhun Neural TTS |
| **Punjabi** | `pa` | 500 | 500 | **1,000** | YouTube Punjabi Speeches & Neural TTS |
| **Odia** | `or` | 500 | 500 | **1,000** | YouTube Odia Broadcasts & Subhasini Neural TTS |
| **Assamese** | `as` | 250 | 250 | **500** | YouTube Assamese Talks & Neural TTS |
| **TOTAL** | | **11,750** | **11,750** | **23,500** | **13 Multilingual YouTube Sources** |

---

## 📌 Benchmark & Quality Standards

- **Exact Duration**: Every single clip formatted strictly to **5.0 seconds** (80,000 audio samples at 16 kHz).
- **Sampling Rate**: Standardized **16 kHz Mono WAV** raw audio.
- **YouTube Human Speech**: Real-world acoustic variance (room reverb, microphone variance, spontaneous conversational cadence, ambient room noise).
- **AI Voices**: Neural speech synthesis with simulated voice codec compression (WhatsApp/Telegram voice note simulation).

---

## 📈 Model Performance (Ensemble Classifier)

- **Dataset Size**: 23,500 audio samples (11,750 YouTube Human, 11,750 Neural AI) across 13 languages
- **Stratified 10-Fold Cross-Validation Accuracy**: **100.00%** (+/- 0.00%)
- **Holdout Test Accuracy (4,700 samples / 20%)**: **100.00%**
- **Top Discriminative Acoustic Features**: `spec_flatness_mean` (0.1353), `spec_bw_std` (0.1288), `mfcc_2_std` (0.1131), `mfcc_3_std` (0.0806), `spec_flatness_std` (0.0681), `spec_cent_std` (0.0519).

---

## 🗂️ Directory Layout

```text
voice-data/
├── voice data/
│   ├── human/           # 11,750 YouTube Human voice 16kHz WAV files (5.0s duration, room ambience)
│   └── ai/              # 11,750 Neural AI voice 16kHz WAV files (5.0s duration, codec simulated)
├── metadata.csv         # Full 23,500-sample metadata index (labels, languages, speakers, transcripts)
├── features.csv         # 52 extracted acoustic features per audio sample
├── fast_extract_23500.py # High-performance multiprocessing feature extraction pipeline
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
python predict.py "voice data/human/human_yt_en_0001.wav"
python predict.py "voice data/ai/ai_yt_en_0001.wav"
```
