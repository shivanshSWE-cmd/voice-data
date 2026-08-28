import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "voice data", "ai")
HUMAN_DIR = os.path.join(BASE_DIR, "voice data", "human")

meta_path = os.path.join(BASE_DIR, "metadata.csv")
df_meta = pd.read_csv(meta_path)
print(f"Loaded metadata.csv with {len(df_meta)} rows.")

TARGET_SR = 16000

features_list = []

print("Extracting 52 acoustic features for 3,000 audio samples...")

for idx, row in df_meta.iterrows():
    fpath = os.path.join(BASE_DIR, row['filepath'].replace('/', os.sep))
    label = row['label']
    
    try:
        y, sr = sf.read(fpath)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            
        # 1. MFCCs (20 coefficients: mean & std)
        mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # 2. Spectral Features
        cent = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # 3. Temporal & Energy Features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # Build feature dictionary
        feat_dict = {
            'filename': row['filename'],
            'label': label,
            'language': row['language'],
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
            feat_dict[f'mfcc_{i+1}_mean'] = float(mfcc_means[i])
            feat_dict[f'mfcc_{i+1}_std'] = float(mfcc_stds[i])
            
        features_list.append(feat_dict)
        
        if (idx + 1) % 500 == 0:
            print(f"[{idx+1}/{len(df_meta)}] Extracted features for {row['filename']}...")
    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")

df_feat = pd.DataFrame(features_list)
feat_path = os.path.join(BASE_DIR, "features.csv")
df_feat.to_csv(feat_path, index=False)

print("\n==========================================================")
print(f"  52 ACOUSTIC FEATURES EXTRACTED FOR {len(df_feat)} AUDIO SAMPLES")
print(f"  Saved to: {feat_path}")
print("==========================================================")
