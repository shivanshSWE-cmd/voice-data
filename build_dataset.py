import os
import io
import asyncio
import tarfile
import requests
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import edge_tts

requests.packages.urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HUMAN_DIR = os.path.join(DATASET_DIR, "human")
AI_DIR = os.path.join(DATASET_DIR, "ai")
TEMP_DIR = os.path.join(BASE_DIR, "temp_downloads")

os.makedirs(HUMAN_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_SR = 16000

# 20 English AI Voices
AI_ENGLISH_VOICES = [
    {"id": "ai_01", "voice": "en-US-AvaNeural", "gender": "Female", "accent": "US", "language": "English", "text": "Artificial intelligence and neural networks are transforming speech recognition technology."},
    {"id": "ai_02", "voice": "en-US-AndrewNeural", "gender": "Male", "accent": "US", "language": "English", "text": "Voice signal processing requires analyzing spectral features to identify acoustic characteristics."},
    {"id": "ai_03", "voice": "en-US-EmmaNeural", "gender": "Female", "accent": "US", "language": "English", "text": "Synthetic speech generation models produce natural pitch contours and rhythmic timing."},
    {"id": "ai_04", "voice": "en-US-BrianNeural", "gender": "Male", "accent": "US", "language": "English", "text": "Deep neural networks are trained on large speech corpora to understand human phonetics."},
    {"id": "ai_05", "voice": "en-US-JennyNeural", "gender": "Female", "accent": "US", "language": "English", "text": "The distinction between synthetic voices and human speech involves micro-temporal variations."},
    {"id": "ai_06", "voice": "en-US-GuyNeural", "gender": "Male", "accent": "US", "language": "English", "text": "Digital filters isolate fundamental frequencies and formant shifts in vocal tracks."},
    {"id": "ai_07", "voice": "en-US-AriaNeural", "gender": "Female", "accent": "US", "language": "English", "text": "Mel frequency cepstral coefficients capture essential spectral envelope properties."},
    {"id": "ai_08", "voice": "en-US-ChristopherNeural", "gender": "Male", "accent": "US", "language": "English", "text": "Supervised machine learning classifiers achieve high precision on audio spectrum features."},
    {"id": "ai_09", "voice": "en-GB-SoniaNeural", "gender": "Female", "accent": "UK", "language": "English", "text": "British English neural models replicate subtle accent intonations and vowel lengthening."},
    {"id": "ai_10", "voice": "en-GB-RyanNeural", "gender": "Male", "accent": "UK", "language": "English", "text": "Zero crossing rates and spectral bandwidth provide strong signals for audio classification."},
    {"id": "ai_11", "voice": "en-GB-MaisieNeural", "gender": "Female", "accent": "UK", "language": "English", "text": "Evaluating model performance requires balanced datasets across multiple speaker demographics."},
    {"id": "ai_12", "voice": "en-GB-ThomasNeural", "gender": "Male", "accent": "UK", "language": "English", "text": "Acoustic feature vectors combine time-domain and frequency-domain statistics."},
    {"id": "ai_13", "voice": "en-AU-NatashaNeural", "gender": "Female", "accent": "AU", "language": "English", "text": "Australian speech synthesis incorporates unique prosodic inflections and speech cadences."},
    {"id": "ai_14", "voice": "en-AU-WilliamNeural", "gender": "Male", "accent": "AU", "language": "English", "text": "Data preprocessing ensures uniform sample rates and amplitude normalization across all files."},
    {"id": "ai_15", "voice": "en-CA-ClaraNeural", "gender": "Female", "accent": "CA", "language": "English", "text": "Canadian accent models broaden the phonetic diversity of synthetic training samples."},
    {"id": "ai_16", "voice": "en-CA-LiamNeural", "gender": "Male", "accent": "CA", "language": "English", "text": "Feature extraction converts raw continuous audio signals into structured numerical matrices."},
    {"id": "ai_17", "voice": "en-IN-NeerjaNeural", "gender": "Female", "accent": "IN", "language": "English", "text": "Indian English neural voices demonstrate distinct syllable pacing and consonant emphasis."},
    {"id": "ai_18", "voice": "en-IN-PrabhatNeural", "gender": "Male", "accent": "IN", "language": "English", "text": "Random forest ensembles combine multiple decision trees for stable decision boundaries."},
    {"id": "ai_19", "voice": "en-IE-EmilyNeural", "gender": "Female", "accent": "IE", "language": "English", "text": "Irish vocal patterns present rich acoustic variation for audio classification benchmarks."},
    {"id": "ai_20", "voice": "en-IE-ConnorNeural", "gender": "Male", "accent": "IE", "language": "English", "text": "Cross-validation validates model robustness against overfitting on training voice features."}
]

# 10 Hindi AI Voices (hi-IN-SwaraNeural & hi-IN-MadhurNeural with varied pitch/rate and text)
AI_HINDI_VOICES = [
    {"id": "ai_hindi_01", "voice": "hi-IN-SwaraNeural", "gender": "Female", "accent": "IN Hindi", "language": "Hindi", "rate": "+0%", "pitch": "+0Hz", "text": "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग तकनीक आधुनिक तकनीक में क्रांति ला रही हैं।"},
    {"id": "ai_hindi_02", "voice": "hi-IN-MadhurNeural", "gender": "Male", "accent": "IN Hindi", "language": "Hindi", "rate": "+0%", "pitch": "+0Hz", "text": "आवाज पहचान प्रणाली अद्वितीय आवाज विशेषताओं की पहचान करने के लिए स्पेक्ट्रल सुविधाओं का विश्लेषण करती है।"},
    {"id": "ai_hindi_03", "voice": "hi-IN-SwaraNeural", "gender": "Female", "accent": "IN Hindi", "language": "Hindi", "rate": "-5%", "pitch": "+5Hz", "text": "सिंथेटिक भाषण उत्पादन मॉडल प्राकृतिक पिच समोच्च और लयबद्ध समय उत्पन्न करते हैं।"},
    {"id": "ai_hindi_04", "voice": "hi-IN-MadhurNeural", "gender": "Male", "accent": "IN Hindi", "language": "Hindi", "rate": "+5%", "pitch": "-5Hz", "text": "डीप लर्निंग मॉडल को प्रभावी प्रशिक्षण और मूल्यांकन के लिए संतुलित डेटासेट की आवश्यकता होती है।"},
    {"id": "ai_hindi_05", "voice": "hi-IN-SwaraNeural", "gender": "Female", "accent": "IN Hindi", "language": "Hindi", "rate": "+5%", "pitch": "-2Hz", "text": "मानव और कृत्रिम आवाज के बीच का अंतर शोध का एक सक्रिय क्षेत्र है।"},
    {"id": "ai_hindi_06", "voice": "hi-IN-MadhurNeural", "gender": "Male", "accent": "IN Hindi", "language": "Hindi", "rate": "-5%", "pitch": "+3Hz", "text": "डिजिटल सिग्नल प्रोसेसिंग ऑडियो सुविधा निष्कर्षण के लिए शक्तिशाली उपकरण प्रदान करता है।"},
    {"id": "ai_hindi_07", "voice": "hi-IN-SwaraNeural", "gender": "Female", "accent": "IN Hindi", "language": "Hindi", "rate": "+0%", "pitch": "+8Hz", "text": "मेल फ्रीक्वेंसी सेपस्ट्रल गुणांक स्पेक्ट्रल लिफाफे के आवश्यक गुणों को कैप्चर करते हैं।"},
    {"id": "ai_hindi_08", "voice": "hi-IN-MadhurNeural", "gender": "Male", "accent": "IN Hindi", "language": "Hindi", "rate": "+2%", "pitch": "-8Hz", "text": "मशीन लर्निंग क्लासिफायर ऑडियो स्पेक्ट्रम सुविधाओं पर उच्च सटीकता प्राप्त कर सकते हैं।"},
    {"id": "ai_hindi_09", "voice": "hi-IN-SwaraNeural", "gender": "Female", "accent": "IN Hindi", "language": "Hindi", "rate": "-2%", "pitch": "-5Hz", "text": "विविध भाषाओं और लहजों में ऑडियो नमूनों का मूल्यांकन मॉडल के प्रदर्शन को सुधारता है।"},
    {"id": "ai_hindi_10", "voice": "hi-IN-MadhurNeural", "gender": "Male", "accent": "IN Hindi", "language": "Hindi", "rate": "+0%", "pitch": "+5Hz", "text": "क्रॉस-वैलिडेशन हमारे भाषण पहचान मॉडल के प्रदर्शन की निरंतरता सुनिश्चित करता है।"}
]

async def generate_single_ai_voice(item):
    filename = f"{item['id']}.wav"
    out_path = os.path.join(AI_DIR, filename)
    temp_mp3 = os.path.join(TEMP_DIR, f"{item['id']}.mp3")
    
    rate = item.get("rate", "+0%")
    pitch = item.get("pitch", "+0Hz")
    
    print(f"Generating AI Voice [{item['id']}] ({item['voice']}, {item['language']})...")
    communicate = edge_tts.Communicate(item["text"], item["voice"], rate=rate, pitch=pitch)
    await communicate.save(temp_mp3)
    
    y, sr = librosa.load(temp_mp3, sr=TARGET_SR, mono=True)
    y = librosa.util.normalize(y)
    
    sf.write(out_path, y, TARGET_SR)
    duration = float(len(y) / TARGET_SR)
    
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
        
    return {
        "filename": filename,
        "filepath": os.path.relpath(out_path, BASE_DIR).replace("\\", "/"),
        "label": "ai",
        "voice_type": "Neural TTS",
        "speaker_id": item["voice"],
        "gender": item["gender"],
        "accent": item["accent"],
        "language": item["language"],
        "duration_sec": round(duration, 2),
        "sample_rate": TARGET_SR,
        "transcript": item["text"]
    }

async def generate_all_ai_voices():
    all_configs = AI_ENGLISH_VOICES + AI_HINDI_VOICES
    tasks = [generate_single_ai_voice(item) for item in all_configs]
    return await asyncio.gather(*tasks)

def fetch_human_english_voices():
    print("\nFetching 20 English Human Voice samples from LibriSpeech dataset...")
    tar_path = os.path.join(TEMP_DIR, "dev-clean-2.tar.gz")
    url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    
    if not os.path.exists(tar_path):
        print(f"Downloading LibriSpeech audio archive (~120MB) to {tar_path}...")
        r = requests.get(url, stream=True, verify=False)
        with open(tar_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
    human_records = []
    found_speakers = {}
    
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".flac"):
                parts = member.name.split("/")
                if len(parts) >= 4:
                    speaker_id = parts[2]
                    if found_speakers.get(speaker_id, 0) < 2 and len(human_records) < 20:
                        found_speakers[speaker_id] = found_speakers.get(speaker_id, 0) + 1
                        
                        f = tar.extractfile(member)
                        if f is not None:
                            idx = len(human_records) + 1
                            filename = f"human_{idx:02d}.wav"
                            out_path = os.path.join(HUMAN_DIR, filename)
                            
                            y, sr = librosa.load(f, sr=TARGET_SR, mono=True)
                            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                            y_normalized = librosa.util.normalize(y_trimmed)
                            
                            max_samples = TARGET_SR * 7
                            if len(y_normalized) > max_samples:
                                y_normalized = y_normalized[:max_samples]
                                
                            sf.write(out_path, y_normalized, TARGET_SR)
                            duration = float(len(y_normalized) / TARGET_SR)
                            
                            f0, _, _ = librosa.pyin(y_normalized, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
                            valid_f0 = f0[~np.isnan(f0)]
                            mean_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else 150
                            inferred_gender = "Female" if mean_f0 > 165 else "Male"
                            
                            human_records.append({
                                "filename": filename,
                                "filepath": os.path.relpath(out_path, BASE_DIR).replace("\\", "/"),
                                "label": "human",
                                "voice_type": "Real Human Recording",
                                "speaker_id": f"LibriSpeech_Spk_{speaker_id}",
                                "gender": inferred_gender,
                                "accent": "US English",
                                "language": "English",
                                "duration_sec": round(duration, 2),
                                "sample_rate": TARGET_SR,
                                "transcript": f"Human English speech sample from speaker {speaker_id}"
                            })
                            print(f"Processed English Human Voice [{filename}] -> Speaker {speaker_id} ({inferred_gender}, {duration:.2f}s)")
                            
                        if len(human_records) >= 20:
                            break
                            
    return human_records

def fetch_human_hindi_voices():
    print("\nFetching 10 Hindi Human Voice samples from OpenSLR 103 Hindi speech corpus...")
    tar_path = os.path.join(TEMP_DIR, "Hindi_test.tar.gz")
    url = "https://www.openslr.org/resources/103/Hindi_test.tar.gz"
    
    if not os.path.exists(tar_path):
        print(f"Downloading OpenSLR Hindi speech archive to {tar_path}...")
        r = requests.get(url, stream=True, verify=False)
        with open(tar_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded {downloaded / (1024*1024):.1f} MB...")
                    if downloaded >= 30 * 1024 * 1024:  # ~30MB provides ample Hindi audio clips
                        break
                        
    human_hindi_records = []
    
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for member in tar:
            if member.name.endswith(".wav"):
                f = tar.extractfile(member)
                if f is not None:
                    idx = len(human_hindi_records) + 1
                    filename = f"human_hindi_{idx:02d}.wav"
                    out_path = os.path.join(HUMAN_DIR, filename)
                    
                    audio_bytes = io.BytesIO(f.read())
                    y, sr = librosa.load(audio_bytes, sr=TARGET_SR, mono=True)
                    
                    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                    y_normalized = librosa.util.normalize(y_trimmed)
                    
                    max_samples = TARGET_SR * 7
                    if len(y_normalized) > max_samples:
                        y_normalized = y_normalized[:max_samples]
                        
                    sf.write(out_path, y_normalized, TARGET_SR)
                    duration = float(len(y_normalized) / TARGET_SR)
                    
                    f0, _, _ = librosa.pyin(y_normalized, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
                    valid_f0 = f0[~np.isnan(f0)]
                    mean_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else 150
                    inferred_gender = "Female" if mean_f0 > 165 else "Male"
                    
                    human_hindi_records.append({
                        "filename": filename,
                        "filepath": os.path.relpath(out_path, BASE_DIR).replace("\\", "/"),
                        "label": "human",
                        "voice_type": "Real Human Recording",
                        "speaker_id": f"OpenSLR_Hindi_{member.name.split('/')[-1].replace('.wav', '')}",
                        "gender": inferred_gender,
                        "accent": "Indian Hindi",
                        "language": "Hindi",
                        "duration_sec": round(duration, 2),
                        "sample_rate": TARGET_SR,
                        "transcript": f"Human Hindi speech utterance {member.name}"
                    })
                    print(f"Processed Hindi Human Voice [{filename}] ({inferred_gender}, {duration:.2f}s)")
                    
                    if len(human_hindi_records) >= 10:
                        break
                        
    return human_hindi_records

def main():
    print("=== Step 1: Generating 30 AI Voice Samples (20 English + 10 Hindi) ===")
    ai_records = asyncio.run(generate_all_ai_voices())
    
    print("\n=== Step 2: Fetching 20 English Human Voice Samples ===")
    human_en_records = fetch_human_english_voices()
    
    print("\n=== Step 3: Fetching 10 Hindi Human Voice Samples ===")
    human_hi_records = fetch_human_hindi_voices()
    
    all_records = human_en_records + human_hi_records + ai_records
    df = pd.DataFrame(all_records)
    
    csv_path = os.path.join(BASE_DIR, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n==========================================")
    print("   EXPANDED DATASET CREATION SUCCESSFUL!")
    print("==========================================")
    print(f"Total Samples: {len(df)}")
    print(f"Human Voices:  {len(df[df['label'] == 'human'])} (20 English + 10 Hindi)")
    print(f"AI Voices:     {len(df[df['label'] == 'ai'])} (20 English + 10 Hindi)")
    print(f"Metadata saved: {csv_path}")

if __name__ == "__main__":
    main()
