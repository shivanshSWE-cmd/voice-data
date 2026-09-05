# 🎙️ Multilingual & Large-Scale English Voice Classification Dataset

A high-capacity dataset for training machine learning and deep learning models to perform **AI vs. Human Voice Detection**. Features **4,000 unique 5-second English audio samples** (**2,000 Human English voices** and **2,000 AI English voices**).

---

## 📊 Dataset Distribution (4,000 English 5-Second Clips)

| Category | Audio Clips | Duration per Clip | Format Specifications | Diversity & Sources |
| :--- | :---: | :---: | :--- | :--- |
| **AI English Voices** | **2,000** | **5.0 seconds** | 16 kHz Mono WAV | 15+ Neural TTS models (Ava, Andrew, Emma, Brian, Sonia, Ryan, Natasha, Neerja, Guy, Jenny, Aria, Clara, Liam) with 2,000 unique sentence prompts, varied speaking rates (-10% to +10%) & pitches (-12Hz to +12Hz) |
| **Human English Voices** | **2,000** | **5.0 seconds** | 16 kHz Mono WAV | LibriSpeech multi-speaker human speech recordings across 100+ male/female speakers with ambient room noise augmentation (SNR 20–30 dB) |
| **TOTAL ENGLISH** | **4,000** | **5.0 seconds** | **16 kHz Mono WAV** | **4,000 Unique 5-Second Audio Files** |

---

## 📌 Benchmark & Quality Standards

- **Exact Duration**: Every single clip formatted strictly to **5.0 seconds** (80,000 audio samples at 16 kHz).
- **Sampling Rate**: Standardized **16 kHz Mono WAV** raw audio.
- **Uniqueness Guarantee**:
  - **AI Voices**: Synthesized using 2,000 distinct sentence prompts across technology, science, conversation, news, and literature.
  - **Human Voices**: Extracted from 2,000 distinct LibriSpeech speaker utterances across male and female speakers.
- **Real-World Acoustic Augmentations**:
  - **Human Voices**: 40% of samples augmented with ambient background room noise, office hum, and mic noise (SNR 20–30 dB).
  - **AI Voices**: 40% of samples augmented with minor voice codec compression simulation (WhatsApp/Telegram voice notes).

---

## 📈 Model Performance (Ensemble Classifier)

- **Dataset Size**: 4,000 English audio samples (2,000 Human, 2,000 AI)
- **Stratified 10-Fold Cross-Validation Accuracy**: **99.60%** (+/- 0.80%)
- **Holdout Test Accuracy (800 samples / 20%)**: **99.38%**
- **Top Discriminative Acoustic Features**: `spec_cent_std`, `mfcc_3_std`, `mfcc_2_std`, `spec_roll_std`, `rms_mean`, `spec_flatness_std`.

---

## 🗂️ Directory Layout

```text
voice-data/
├── voice data/
│   ├── human/           # 2,000 Human English 16kHz WAV files (5.0s duration, ambient noise)
│   └── ai/              # 2,000 Neural AI English 16kHz WAV files (5.0s duration, codec simulated)
├── metadata.csv         # Full 4,000-sample metadata index (labels, languages, speakers, transcripts)
├── features.csv         # 52 extracted acoustic features per audio sample
├── build_4000_english_dataset.py # English dataset generator script
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
python predict.py "voice data/human/human_en_0001.wav"
python predict.py "voice data/ai/ai_en_0001.wav"
```
