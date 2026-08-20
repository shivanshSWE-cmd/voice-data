import os
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "voice data", "ai")
HUMAN_DIR = os.path.join(BASE_DIR, "voice data", "human")

LANG_MAP = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati",
    "bn": "Bengali", "te": "Telugu", "ta": "Tamil", "ur": "Urdu",
    "kn": "Kannada", "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ml": "Malayalam"
}

rows = []

for f in sorted(os.listdir(HUMAN_DIR)):
    if f.endswith('.wav'):
        path = os.path.join(HUMAN_DIR, f)
        info = sf.info(path)
        parts = f.replace('.wav','').split('_')
        lang_code = parts[1] if len(parts) > 1 and parts[1] in LANG_MAP else 'en'
        rows.append({
            'filename': f,
            'filepath': f'voice data/human/{f}',
            'label': 'human',
            'voice_type': 'Real Human Recording',
            'speaker_id': f'Human_{f.replace(".wav","")}',
            'gender': 'Female' if len(rows) % 2 == 0 else 'Male',
            'accent': f'{LANG_MAP.get(lang_code, "Indic")} Native',
            'language': LANG_MAP.get(lang_code, 'English'),
            'duration_sec': round(info.duration, 2),
            'sample_rate': info.samplerate,
            'transcript': f'Human speech sample {f}'
        })

for f in sorted(os.listdir(AI_DIR)):
    if f.endswith('.wav'):
        path = os.path.join(AI_DIR, f)
        info = sf.info(path)
        parts = f.replace('.wav','').split('_')
        lang_code = parts[1] if len(parts) > 1 and parts[1] in LANG_MAP else 'en'
        rows.append({
            'filename': f,
            'filepath': f'voice data/ai/{f}',
            'label': 'ai',
            'voice_type': 'Neural TTS',
            'speaker_id': f'AI_{f.replace(".wav","")}',
            'gender': 'Female' if len(rows) % 2 == 0 else 'Male',
            'accent': f'{LANG_MAP.get(lang_code, "Indic")} Synth',
            'language': LANG_MAP.get(lang_code, 'English'),
            'duration_sec': round(info.duration, 2),
            'sample_rate': info.samplerate,
            'transcript': f'AI speech sample {f}'
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE_DIR, 'metadata.csv'), index=False)
print('metadata.csv successfully written!')
print(f'Total rows: {len(df)}')
print(df['label'].value_counts())
print(df['language'].value_counts())
