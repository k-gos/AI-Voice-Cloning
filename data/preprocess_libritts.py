import os
import pandas as pd
import soundfile as sf
import numpy as np

RAW_ROOT = "data/raw/libritts/LibriTTS"
PROCESSED_ROOT = "data/processed"
METADATA_OUT = os.path.join(PROCESSED_ROOT, "metadata.csv")

splits = [
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other"
]

# Load existing metadata if cache file exists
if os.path.exists(METADATA_OUT):
    df_cached = pd.read_csv(METADATA_OUT)
    cached_paths = set(df_cached["audio_path"].tolist())
    print(f"Loaded {len(cached_paths)} cached entries.")
else:
    df_cached = pd.DataFrame()
    cached_paths = set()
    print("No existing metadata found. Starting fresh.")

new_metadata = []

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

            trans_files = [f for f in os.listdir(chapter_dir) if f.endswith('.trans.tsv')]
            for trans_file in trans_files:
                trans_path = os.path.join(chapter_dir, trans_file)

                try:
                    df = pd.read_csv(trans_path, sep='\t', comment='#', header=None, names=['utt_id', 'original', 'normalized'])
                except Exception as e:
                    print(f"Error reading {trans_path}: {e}")
                    continue

                for _, row in df.iterrows():
                    utt_id = row['utt_id']
                    text = str(row['normalized']).strip()

                    # Skip missing or empty text
                    if not text or text.lower() == 'nan':
                        continue

                    wav_file = f"{utt_id}.wav"
                    wav_path = os.path.join(chapter_dir, wav_file)

                    if not os.path.exists(wav_path):
                        continue

                    # Compute relative path for comparison
                    rel_audio_path = os.path.relpath(wav_path, PROCESSED_ROOT).replace("\\", "/")

                    # Skip if already cached
                    if rel_audio_path in cached_paths:
                        continue

                    try:
                        duration = sf.info(wav_path).duration
                    except Exception as e:
                        print(f"Error reading {wav_path}: {e}")
                        continue

                    if split.startswith("train-"):
                        split_value = "train"
                    elif split.startswith("dev-"):
                        split_value = "val"
                    elif split.startswith("test-"):
                        split_value = "test"
                    else:
                        split_value = split

                    new_metadata.append({
                        "audio_path": rel_audio_path,
                        "text": text,
                        "speaker_id": speaker,
                        "emotion": "neutral",
                        "duration": duration,
                        "split": split_value
                    })

# Create new DataFrame and clean it
df_new = pd.DataFrame(new_metadata)
df_new = df_new.dropna(subset=["text"])
df_new = df_new[df_new["text"].str.strip().astype(bool)]

# Combine with cached data
df_combined = pd.concat([df_cached, df_new], ignore_index=True)

# If no val samples, assign 10% of train to val
if (df_combined['split'] == 'val').sum() == 0:
    train_idx = df_combined[df_combined['split'] == 'train'].index
    val_size = int(0.1 * len(train_idx))
    val_idx = np.random.choice(train_idx, size=val_size, replace=False)
    df_combined.loc[val_idx, 'split'] = 'val'
    print(f"Randomly assigned {val_size} samples from train to val split.")

# Save
os.makedirs(PROCESSED_ROOT, exist_ok=True)
df_combined.to_csv(METADATA_OUT, index=False)
print(f"Saved metadata with {len(df_combined)} total entries to {METADATA_OUT}")
