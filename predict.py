import os
import sys
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_classifier.pkl")
TARGET_SR = 16000

def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run train_baseline.py first.")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload['model'], payload['scaler'], payload['feature_names']

def extract_audio_features(audio_path, expected_features):
    y, sr = sf.read(audio_path)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        
    mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    
    cent = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    feat = {
        'spec_cent_mean': float(np.mean(cent)),
        'spec_cent_std': float(np.std(cent)),
        'spec_bw_mean': float(np.mean(bw)),
        'spec_bw_std': float(np.std(bw)),
        'spec_roll_mean': float(np.mean(rolloff)),
        'spec_roll_std': float(np.std(rolloff)),
        'spec_flatness_mean': float(np.mean(flatness)),
        'spec_flatness_std': float(np.std(flatness)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms))
    }
    
    for i in range(20):
        feat[f'mfcc_{i+1}_mean'] = float(mfcc_means[i])
        feat[f'mfcc_{i+1}_std'] = float(mfcc_stds[i])
        
    df_feat = pd.DataFrame([feat])
    return df_feat[expected_features]

def predict_voice(audio_path):
    print(f"\nAnalyzing audio file: {audio_path}")
    model, scaler, feature_names = load_classifier()
    X_single = extract_audio_features(audio_path, feature_names)
    X_scaled = scaler.transform(X_single)
    
    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    classes = model.classes_
    
    prob_dict = dict(zip(classes, probs))
    human_conf = prob_dict.get('human', 0.0) * 100
    ai_conf = prob_dict.get('ai', 0.0) * 100
    
    print("=" * 42)
    print(f" PREDICTION RESULT: {pred.upper()} VOICE")
    print("=" * 42)
    print(" Confidence:")
    print(f"   - Human Voice: {human_conf:.2f}%")
    print(f"   - AI Voice:    {ai_conf:.2f}%")
    print("=" * 42)
    return pred, human_conf, ai_conf

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_voice(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_audio_file>")
