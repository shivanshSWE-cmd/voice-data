import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(BASE_DIR, "metadata.csv")
df_meta = pd.read_csv(meta_path)
print(f"Extracting 52 acoustic features for {len(df_meta)} audio samples...")

TARGET_SR = 16000

def extract_row_features(item):
    idx, row = item
    fpath = os.path.join(BASE_DIR, row['filepath'].replace('/', os.sep))
    label = row['label']
    
    try:
        y, sr = sf.read(fpath)
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
            feat[f'mfcc_{i+1}_mean'] = float(mfcc_means[i])
            feat[f'mfcc_{i+1}_std'] = float(mfcc_stds[i])
            
        return idx, feat
    except Exception as e:
        return idx, None

def main():
    results = [None] * len(df_meta)
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(extract_row_features, (i, r)) for i, r in df_meta.iterrows()]
        done = 0
        for f in as_completed(futures):
            idx, res = f.result()
            if res: results[idx] = res
            done += 1
            if done % 500 == 0:
                print(f"  [{done}/{len(df_meta)}] Extracted acoustic features...")
                
    valid = [r for r in results if r is not None]
    df_feat = pd.DataFrame(valid)
    feat_path = os.path.join(BASE_DIR, "features.csv")
    df_feat.to_csv(feat_path, index=False)
    print(f"Extracted 52 acoustic features for {len(df_feat)} samples -> {feat_path}")

if __name__ == "__main__":
    main()
