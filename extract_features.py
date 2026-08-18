import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "features.csv")

def extract_audio_features(filepath):
    """
    Extracts comprehensive acoustic features from an audio file:
    - 13 MFCCs (mean & std) -> 26 features
    - Spectral Centroid (mean & std) -> 2 features
    - Spectral Bandwidth (mean & std) -> 2 features
    - Spectral Rolloff (mean & std) -> 2 features
    - Zero Crossing Rate (mean & std) -> 2 features
    - RMS Energy (mean & std) -> 2 features
    - Pitch / Fundamental Frequency (mean & std) -> 2 features
    Total: 38 acoustic features
    """
    full_path = os.path.join(BASE_DIR, filepath)
    y, sr = librosa.load(full_path, sr=16000, mono=True)
    
    # 1. MFCCs (13 coefficients)
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
    
    # 7. Pitch (f0)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    valid_f0 = f0[~np.isnan(f0)]
    pitch_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
    
    features = {}
    
    # Add MFCC features
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc_mean[i]
        features[f"mfcc_{i+1}_std"] = mfcc_std[i]
        
    features["spec_cent_mean"] = spec_cent_mean
    features["spec_cent_std"] = spec_cent_std
    features["spec_bw_mean"] = spec_bw_mean
    features["spec_bw_std"] = spec_bw_std
    features["spec_roll_mean"] = spec_roll_mean
    features["spec_roll_std"] = spec_roll_std
    features["zcr_mean"] = zcr_mean
    features["zcr_std"] = zcr_std
    features["rms_mean"] = rms_mean
    features["rms_std"] = rms_std
    features["pitch_mean"] = pitch_mean
    features["pitch_std"] = pitch_std
    
    return features

def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found. Please run build_dataset.py first.")
        return
        
    df_meta = pd.read_csv(METADATA_PATH)
    print(f"Extracting features for {len(df_meta)} audio samples...")
    
    extracted_rows = []
    for idx, row in df_meta.iterrows():
        print(f"[{idx+1}/{len(df_meta)}] Extracting features from {row['filename']} ({row['label']})...")
        feats = extract_audio_features(row['filepath'])
        feats['filename'] = row['filename']
        feats['label'] = row['label']
        extracted_rows.append(feats)
        
    df_feats = pd.DataFrame(extracted_rows)
    
    # Reorder columns so filename and label come first
    cols = ['filename', 'label'] + [c for c in df_feats.columns if c not in ['filename', 'label']]
    df_feats = df_feats[cols]
    
    df_feats.to_csv(FEATURES_PATH, index=False)
    print(f"\nExtracted {len(df_feats.columns)-2} features for {len(df_feats)} audio files.")
    print(f"Features saved to: {FEATURES_PATH}")

if __name__ == "__main__":
    main()
