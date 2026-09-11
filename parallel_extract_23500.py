import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
OUT_FEAT_PATH = os.path.join(BASE_DIR, "features.csv")
TARGET_SR = 16000

df_meta = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_meta)} rows from metadata.csv.")

def process_single(row):
    rel_path = row['filepath']
    audio_full_path = os.path.join(BASE_DIR, rel_path)
    
    try:
        y, sr = sf.read(audio_full_path)
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
            'filename': row['filename'],
            'label': row['label'],
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
            feat[f'mfcc_{i+1}_mean'] = float(mfcc_means[i])
            feat[f'mfcc_{i+1}_std'] = float(mfcc_stds[i])
            
        return feat
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return None

rows_list = df_meta.to_dict('records')

print(f"Starting multi-threaded feature extraction for {len(rows_list)} samples...")
extracted_features = []

with ThreadPoolExecutor(max_workers=16) as executor:
    results = executor.map(process_single, rows_list)
    for idx, r in enumerate(results):
        if r:
            extracted_features.append(r)
        if (idx + 1) % 2500 == 0:
            print(f"  [{idx+1}/{len(rows_list)}] Extracted features...")

df_features = pd.DataFrame(extracted_features)
df_features.to_csv(OUT_FEAT_PATH, index=False)

print("\n==========================================================")
print(f"  PARALLEL FEATURE EXTRACTION COMPLETE: {len(df_features)} ROWS")
print(f"  Saved to: {OUT_FEAT_PATH}")
print("==========================================================")
