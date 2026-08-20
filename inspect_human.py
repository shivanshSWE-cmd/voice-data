import os
import numpy as np
import soundfile as sf
import librosa

HUMAN_DIR = r"d:\Github projects\voice_dataset\voice data\human"
files = sorted([f for f in os.listdir(HUMAN_DIR) if f.endswith('.wav')])
print(f"Total human files: {len(files)}")

stats = []
for f in files:
    path = os.path.join(HUMAN_DIR, f)
    try:
        y, sr = sf.read(path)
        rms = float(np.sqrt(np.mean(y**2)))
        max_val = float(np.max(np.abs(y)))
        zero_ratio = float(np.mean(np.abs(y) < 1e-4))
        dur = float(len(y) / sr)
        stats.append((f, dur, max_val, rms, zero_ratio))
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\nSample Human File Audio Stats (First 25):")
print(f"{'Filename':<22} | {'Dur':<6} | {'MaxAmp':<8} | {'RMS':<8} | {'Silence%':<8}")
print("-" * 60)
for s in stats[:25]:
    print(f"{s[0]:<22} | {s[1]:<6.2f} | {s[2]:<8.4f} | {s[3]:<8.4f} | {s[4]*100:<8.1f}")

max_amps = [s[2] for s in stats]
rms_vals = [s[3] for s in stats]
durs = [s[1] for s in stats]
print(f"\nStats summary across all {len(stats)} human files:")
print(f"  - Min Max-Amp: {min(max_amps):.4f}, Max Max-Amp: {max(max_amps):.4f}")
print(f"  - Min RMS:     {min(rms_vals):.4f}, Max RMS:     {max(rms_vals):.4f}")
print(f"  - Min Duration:{min(durs):.2f}s, Max Duration:{max(durs):.2f}s")

quiet = [s for s in stats if s[3] < 0.02 or s[2] < 0.1]
print(f"\nQuiet or low-quality files count: {len(quiet)}")
if quiet:
    print("Quiet files sample:", [q[0] for q in quiet[:15]])
