import os
import pandas as pd
import soundfile as sf

RAW_ROOT = "data/raw/libritts/LibriTTS"
PROCESSED_ROOT = "data/processed"
METADATA_OUT = os.path.join(PROCESSED_ROOT, "metadata.csv")

splits = [
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other"
]

metadata = []

for split in splits:
    split_dir = os.path.join(RAW_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    for speaker in os.listdir(split_dir):
        speaker_dir = os.path.join(split_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if not os.path.isdir(chapter_dir):
                continue
            # Find the .trans.tsv file(s)
            trans_files = [f for f in os.listdir(chapter_dir) if f.endswith('.trans.tsv')]
            for trans_file in trans_files:
                trans_path = os.path.join(chapter_dir, trans_file)
                # Read the tsv, skip comments
                df = pd.read_csv(trans_path, sep='\t', comment='#', header=None, names=['utt_id', 'original', 'normalized'])
                for _, row in df.iterrows():
                    utt_id = row['utt_id']
                    text = str(row['normalized']).strip()
                    wav_file = f"{utt_id}.wav"
                    wav_path = os.path.join(chapter_dir, wav_file)
                    if not os.path.exists(wav_path):
                        continue
                    # Get duration
                    duration = sf.info(wav_path).duration
                    # Relative path for metadata
                    rel_audio_path = os.path.relpath(wav_path, PROCESSED_ROOT).replace("\\", "/")
                    # Set split value for compatibility
                    if split.startswith("train-"):
                        split_value = "train"
                    elif split.startswith("dev-"):
                        split_value = "val"
                    elif split.startswith("test-"):
                        split_value = "test"
                    else:
                        split_value = split
                    # Add to metadata
                    metadata.append({
                        "audio_path": rel_audio_path,
                        "text": text,
                        "speaker_id": speaker,
                        "emotion": "neutral",
                        "duration": duration,
                        "split": split_value
                    })

# Save metadata
os.makedirs(PROCESSED_ROOT, exist_ok=True)
df_out = pd.DataFrame(metadata)
df_out.to_csv(METADATA_OUT, index=False)
print(f"Saved metadata with {len(df_out)} entries to {METADATA_OUT}") 