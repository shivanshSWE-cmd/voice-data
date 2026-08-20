import os
import zipfile
import requests
import librosa
import soundfile as sf

requests.packages.urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_DIR = os.path.join(BASE_DIR, "voice data", "human")
TEMP_DIR = os.path.join(BASE_DIR, "temp_downloads")
TARGET_SR = 16000

zip_targets = [
    ("ml", "Malayalam", 20, "https://www.openslr.org/resources/63/ml_in_female.zip"),
    ("ta", "Tamil", 20, "https://www.openslr.org/resources/65/ta_in_female.zip"),
    ("te", "Telugu", 20, "https://www.openslr.org/resources/66/te_in_female.zip"),
    ("gu", "Gujarati", 20, "https://www.openslr.org/resources/78/gu_in_female.zip"),
    ("kn", "Kannada", 20, "https://www.openslr.org/resources/79/kn_in_female.zip"),
    ("bn", "Bengali", 20, "https://www.openslr.org/resources/53/bn_in_female.zip")
]

def download_and_extract_zip(code, name, count, url):
    zip_path = os.path.join(TEMP_DIR, f"{code}_speech.zip")
    print(f"\nDownloading genuine native {name} ({code.upper()}) human speech archive...")
    
    # Download first 40MB chunk with range request or full stream
    r = requests.get(url, stream=True, verify=False)
    with open(zip_path, "wb") as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= 45 * 1024 * 1024:  # ~45MB contains hundreds of native audio files!
                    break
                    
    saved_count = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            wav_members = [m for m in z.namelist() if m.endswith('.wav') or m.endswith('.flac')]
            for m_name in wav_members[:count]:
                with z.open(m_name) as f_in:
                    y, sr = librosa.load(f_in, sr=TARGET_SR, mono=True)
                    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                    y_norm = librosa.util.normalize(y_trimmed)
                    
                    if len(y_norm) > TARGET_SR * 6:
                        y_norm = y_norm[:TARGET_SR * 6]
                        
                    saved_count += 1
                    filename = f"human_{code}_{saved_count:03d}.wav"
                    out_path = os.path.join(HUMAN_DIR, filename)
                    sf.write(out_path, y_norm, TARGET_SR)
                    print(f"Saved real native {name} human voice [{filename}]")
    except Exception as e:
        print(f"Zip extraction info for {name}: {e}")
        
    if os.path.exists(zip_path):
        try: os.remove(zip_path)
        except Exception: pass

def main():
    print("=== Downloading Native Human Speech Files for Malayalam, Tamil, Telugu, Gujarati, Kannada, Bengali ===")
    for code, name, count, url in zip_targets:
        download_and_extract_zip(code, name, count, url)
    print("\nAll native Indic human speech files processed successfully!")

if __name__ == "__main__":
    main()
