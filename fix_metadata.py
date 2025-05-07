import pandas as pd
from pathlib import Path

# Path to your metadata.csv
metadata_path = Path("data/processed/metadata.csv")
audio_root = Path("data/processed/audio/LibriTTS")

# Load metadata
df = pd.read_csv(metadata_path)

# Remove rows with missing or empty text
df = df[df['text'].notna() & (df['text'].str.strip() != '')]

# Fix audio_path to be relative to metadata.csv
def fix_audio_path(row):
    # Extract the filename from the original path
    filename = Path(row['audio_path']).name
    # Extract speaker and chapter from the original path
    parts = Path(row['audio_path']).parts
    # Find the index of 'train-clean-100' or similar
    for i, part in enumerate(parts):
        if part.startswith('train-clean'):
            speaker = parts[i+1]
            chapter = parts[i+2]
            break
    else:
        # If not found, return as is
        return row['audio_path']
    # Build new relative path
    return f"audio/LibriTTS/{parts[i]}/{speaker}/{chapter}/{filename}"

df['audio_path'] = df.apply(fix_audio_path, axis=1)

# Save cleaned metadata
cleaned_path = metadata_path.parent / "metadata_clean.csv"
df.to_csv(cleaned_path, index=False)
print(f"Cleaned metadata saved to {cleaned_path}")