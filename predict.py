import os
import sys
import joblib
import librosa
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_classifier.pkl")

def extract_single_audio_features(filepath, expected_feature_names):
    y, sr = librosa.load(filepath, sr=16000, mono=True)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    valid_f0 = f0[~np.isnan(f0)]
    pitch_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
    
    features = {}
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc_mean[i]
        features[f"mfcc_{i+1}_std"] = mfcc_std[i]
        
    features["spec_cent_mean"] = np.mean(spec_cent)
    features["spec_cent_std"] = np.std(spec_cent)
    features["spec_bw_mean"] = np.mean(spec_bw)
    features["spec_bw_std"] = np.std(spec_bw)
    features["spec_roll_mean"] = np.mean(spec_roll)
    features["spec_roll_std"] = np.std(spec_roll)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    features["pitch_mean"] = pitch_mean
    features["pitch_std"] = pitch_std
    
    df_feat = pd.DataFrame([features])
    return df_feat[expected_feature_names]

def predict_voice(audio_file_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Please run train_baseline.py first.")
        return
        
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
        
    payload = joblib.load(MODEL_PATH)
    model = payload['model']
    scaler = payload['scaler']
    feature_names = payload['feature_names']
    
    print(f"\nAnalyzing audio file: {audio_file_path}")
    X_single = extract_single_audio_features(audio_file_path, feature_names)
    X_scaled = scaler.transform(X_single)
    
    pred_label = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    classes = model.classes_
    
    prob_dict = dict(zip(classes, probs))
    
    print("==========================================")
    print(f" PREDICTION RESULT: {pred_label.upper()} VOICE")
    print("==========================================")
    print(f" Confidence:")
    print(f"   - Human Voice: {prob_dict.get('human', 0)*100:.2f}%")
    print(f"   - AI Voice:    {prob_dict.get('ai', 0)*100:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_voice(sys.argv[1])
    else:
        test_human = os.path.join(BASE_DIR, "voice data", "human", "human_01.wav")
        test_ai = os.path.join(BASE_DIR, "voice data", "ai", "ai_01.wav")
        
        if os.path.exists(test_human):
            predict_voice(test_human)
        if os.path.exists(test_ai):
            predict_voice(test_ai)
