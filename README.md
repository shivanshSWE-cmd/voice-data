# 🎙️ Multilingual YouTube Human & Neural AI Voice Dataset (2,900 Samples)

A production-grade dataset for training machine learning and deep learning models to perform **AI vs. Real-World Human Voice Detection** across **13 major languages**, with human audio clips extracted from **authentic YouTube speech streams** (podcasts, speeches, interviews, broadcasts).

---

## 📊 Dataset Distribution Across 13 Languages (2,900 Audio Samples)

| Language Name | Language Code | YouTube Human Clips | Neural AI Clips | Total Audio Files | Source Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **English** | `en` | 300 | 300 | **600** | YouTube Speech Streams & Multi-Speaker Neural TTS |
| **Hindi** | `hi` | 300 | 300 | **600** | YouTube Hindi Podcasts & Swara/Madhur Neural TTS |
| **Marathi** | `mr` | 150 | 150 | **300** | YouTube Marathi Broadcasts & Manohar Neural TTS |
| **Bengali** | `bn` | 100 | 100 | **200** | YouTube Bengali Speeches & Bashkar Neural TTS |
| **Telugu** | `te` | 100 | 100 | **200** | YouTube Telugu Talks & Mohan Neural TTS |
| **Tamil** | `ta` | 100 | 100 | **200** | YouTube Tamil Discussions & Valluvar Neural TTS |
| **Gujarati** | `gu` | 75 | 75 | **150** | YouTube Gujarati Talks & Niranjan Neural TTS |
| **Kannada** | `kn` | 75 | 75 | **150** | YouTube Kannada Broadcasts & Gagan Neural TTS |
| **Malayalam** | `ml` | 75 | 75 | **150** | YouTube Malayalam Podcasts & Midhun Neural TTS |
| **Punjabi** | `pa` | 50 | 50 | **100** | YouTube Punjabi Speeches & Neural TTS |
| **Urdu** | `ur` | 50 | 50 | **100** | YouTube Urdu Discussions & Salman Neural TTS |
| **Odia** | `or` | 50 | 50 | **100** | YouTube Odia Broadcasts & Subhasini Neural TTS |
| **Assamese** | `as` | 25 | 25 | **50** | YouTube Assamese Talks & Neural TTS |
| **TOTAL** | | **1,450** | **1,450** | **2,900** | **13 Multilingual YouTube Sources** |

---

## 📌 Benchmark & Quality Standards

- **Exact Duration**: Every single clip formatted strictly to **5.0 seconds** (80,000 audio samples at 16 kHz).
- **Sampling Rate**: Standardized **16 kHz Mono WAV** raw audio.
- **YouTube Human Speech**: Real-world acoustic variance (room reverb, microphone variance, spontaneous conversational cadence, ambient room noise).
- **AI Voices**: Neural speech synthesis with simulated voice codec compression (WhatsApp/Telegram voice note simulation).

---

## 📈 Model Performance (Ensemble Classifier)

- **Dataset Size**: 2,900 audio samples (1,450 YouTube Human, 1,450 Neural AI)
- **Stratified 10-Fold Cross-Validation Accuracy**: **100.00%** (+/- 0.00%)
- **Holdout Test Accuracy (580 samples / 20%)**: **100.00%**
- **Top Discriminative Acoustic Features**: `spec_flatness_mean`, `mfcc_2_std`, `spec_bw_std`, `spec_flatness_std`, `mfcc_9_mean`, `rms_mean`.

---

## 🗂️ Directory Layout

```text
voice-data/
├── voice data/
│   ├── human/           # 1,450 YouTube Human voice 16kHz WAV files (5.0s duration, room ambience)
│   └── ai/              # 1,450 Neural AI voice 16kHz WAV files (5.0s duration, codec simulated)
├── metadata.csv         # Full 2,900-sample metadata index (labels, languages, speakers, transcripts)
├── features.csv         # 52 extracted acoustic features per audio sample
├── build_youtube_human_dataset.py # YouTube human voice extraction pipeline
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
python predict.py "voice data/human/human_yt_en_0001.wav"
python predict.py "voice data/ai/ai_yt_en_0001.wav"
```
