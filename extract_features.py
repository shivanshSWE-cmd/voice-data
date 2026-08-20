import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "features.csv")

def extract_audio_features(filepath):
    full_path = os.path.join(BASE_DIR, filepath)
    y, sr = librosa.load(full_path, sr=16000, mono=True)
    
    # 1. MFCCs (13 coefficients: mean & std) -> 26 features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 2. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_std = np.std(spec_cent)
    
    # 3. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_bw_mean = np.mean(spec_bw)
    spec_bw_std = np.std(spec_bw)
    
    # 4. Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_roll_mean = np.mean(spec_roll)
    spec_roll_std = np.std(spec_roll)
    
    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 7. Fast Pitch Proxy via Spectral Flatness & Autocorrelation
    spec_flat = librosa.feature.spectral_flatness(y=y)[0]
    pitch_mean = np.mean(spec_flat)
    pitch_std = np.std(spec_flat)
    
    features = {}
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(mfcc_mean[i])
        features[f"mfcc_{i+1}_std"] = float(mfcc_std[i])
        
    features["spec_cent_mean"] = float(spec_cent_mean)
    features["spec_cent_std"] = float(spec_cent_std)
    features["spec_bw_mean"] = float(spec_bw_mean)
    features["spec_bw_std"] = float(spec_bw_std)
    features["spec_roll_mean"] = float(spec_roll_mean)
    features["spec_roll_std"] = float(spec_roll_std)
    features["zcr_mean"] = float(zcr_mean)
    features["zcr_std"] = float(zcr_std)
    features["rms_mean"] = float(rms_mean)
    features["rms_std"] = float(rms_std)
    features["pitch_mean"] = float(pitch_mean)
    features["pitch_std"] = float(pitch_std)
    
    return features

def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return
        
    df_meta = pd.read_csv(METADATA_PATH)
    print(f"Extracting features for {len(df_meta)} audio samples...")
    
    extracted_rows = []
    for idx, row in df_meta.iterrows():
        if (idx + 1) % 100 == 0 or idx == len(df_meta) - 1:
            print(f"[{idx+1}/{len(df_meta)}] Extracting features from {row['filename']} ({row['label']})...")
        feats = extract_audio_features(row['filepath'])
        feats['filename'] = row['filename']
        feats['label'] = row['label']
        extracted_rows.append(feats)
        
    df_feats = pd.DataFrame(extracted_rows)
    cols = ['filename', 'label'] + [c for c in df_feats.columns if c not in ['filename', 'label']]
    df_feats = df_feats[cols]
    
    df_feats.to_csv(FEATURES_PATH, index=False)
    print(f"\nExtracted {len(df_feats.columns)-2} features for {len(df_feats)} audio files.")
    print(f"Features saved to: {FEATURES_PATH}")

if __name__ == "__main__":
    main()
