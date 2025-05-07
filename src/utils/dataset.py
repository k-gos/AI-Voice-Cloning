import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import yaml
import json
from tqdm import tqdm
import librosa
from phonemizer import phonemize
import re
import logging

logger = logging.getLogger(__name__)

class VoiceCloningDataset(Dataset):
    """Dataset for voice cloning"""
    
    def __init__(
        self,
        metadata_path: str,
        config_path: str,
        split: str = 'train',
        max_audio_len: Optional[int] = None,
        use_cache: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            metadata_path: Path to metadata CSV file
            config_path: Path to config YAML file
            split: Dataset split (train/val/test)
            max_audio_len: Maximum audio length in samples
            use_cache: Whether to cache audio files
        """
        self.root_path = Path(metadata_path).parent
        logger.info(f"Dataset root path: {self.root_path}")
        
        # Load config first to get emotions list
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup emotion parameters before loading metadata
        self.emotions = self.config.get('emotion', {}).get('emotions', ['neutral'])
        if not self.emotions:
            self.emotions = ['neutral']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        logger.info(f"Using emotions: {self.emotions}")
        
        # Load metadata with explicit dtypes
        logger.info(f"Loading metadata from: {metadata_path}")
        self.metadata = pd.read_csv(
            metadata_path,
            dtype={
                'text': str,
                'audio_path': str,
                'speaker_id': str,
                'emotion': str,
                'split': str,
                'duration': float
            }
        )
        logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        
        # Print first few rows for debugging
        logger.info("\nFirst few metadata entries:")
        logger.info(self.metadata.head())
        
        # Convert relative audio paths to absolute paths
        def convert_audio_path(audio_path):
            if pd.isna(audio_path):
                return None
            
            # If it's already an absolute path and exists, use it
            if os.path.isabs(audio_path) and os.path.exists(audio_path):
                return audio_path
                
            # Try different path combinations
            possible_paths = [
                self.root_path / audio_path,  # Relative to metadata.csv location
                self.root_path / 'audio' / audio_path,  # Under audio subdirectory
                Path(audio_path),  # As is
                Path('audio') / audio_path,  # Under audio directory relative to current
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.debug(f"Found audio file at: {path}")
                    return str(path)
            
            logger.debug(f"Could not find audio file for path: {audio_path}")
            return None
        
        logger.info("Converting audio paths...")
        self.metadata['audio_path'] = self.metadata['audio_path'].apply(convert_audio_path)
        
        # Handle missing text values
        if self.metadata['text'].isna().any():
            missing_text_count = self.metadata['text'].isna().sum()
            logger.warning(f"Found {missing_text_count} missing values in column 'text'. Filling with empty string.")
            self.metadata['text'] = self.metadata['text'].fillna('')
        
        # Clean metadata
        self._clean_metadata()
        
        # Filter by split
        split_counts = self.metadata['split'].value_counts()
        logger.info(f"\nSplit distribution before filtering:\n{split_counts}")
        
        self.metadata = self.metadata[self.metadata['split'] == split]
        logger.info(f"After filtering for '{split}' split: {len(self.metadata)} samples")
        
        if len(self.metadata) == 0:
            logger.error(f"No samples found for split '{split}'")
            logger.error("Please check:")
            logger.error("1. Your metadata.csv has the correct 'split' values")
            logger.error("2. Audio files exist in the correct location")
            logger.error("3. The paths in metadata.csv match your audio file structure")
            raise ValueError(f"No samples found for split '{split}'")
        
        # Setup audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_fft = self.config['audio']['n_fft']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio']['fmin']
        self.fmax = self.config['audio']['fmax']
        
        # Setup cache
        self.use_cache = use_cache
        self.cache_dir = self.root_path / 'cache' / split
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup max length
        self.max_audio_len = max_audio_len or self.config['dataset'].get('max_audio_len', None)
        
        # Setup text processing
        self.vocab_size = self.config['model']['text_encoder']['vocab_size']
        self.max_text_len = self.config['dataset'].get('max_text_len', 100)
        
        logger.info(f"Successfully initialized {split} dataset with {len(self.metadata)} samples")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ''
            
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _clean_metadata(self):
        """Clean metadata by removing invalid samples"""
        initial_len = len(self.metadata)
        
        # Verify audio paths exist
        valid_audio = self.metadata['audio_path'].apply(lambda x: os.path.exists(x))
        
        # Verify emotions are valid
        valid_emotions = self.metadata['emotion'].isin(self.emotions)
        
        # Verify speaker IDs are not empty
        valid_speakers = self.metadata['speaker_id'].notna()
        
        # Verify durations are positive
        valid_durations = self.metadata['duration'] > 0
        
        # Apply all filters
        self.metadata = self.metadata[valid_audio & valid_emotions & valid_speakers & valid_durations]
        
        removed = initial_len - len(self.metadata)
        if removed > 0:
            logger.warning(f"Removed {removed} invalid samples from metadata")
            logger.warning(f"- Missing audio files: {(~valid_audio).sum()}")
            logger.warning(f"- Invalid emotions: {(~valid_emotions).sum()}")
            logger.warning(f"- Invalid speaker IDs: {(~valid_speakers).sum()}")
            logger.warning(f"- Invalid durations: {(~valid_durations).sum()}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        # Ensure text is string and clean
        text = self._clean_text(text)
            
        # Simple character-level tokenization
        chars = list(text)
        # Convert characters to indices (0-255 for ASCII)
        indices = [ord(c) % self.vocab_size for c in chars]
        # Pad or truncate to max_text_len
        if len(indices) > self.max_text_len:
            indices = indices[:self.max_text_len]
        else:
            indices = indices + [0] * (self.max_text_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = Path(row['audio_path'])
        if self.use_cache:
            cache_path = self.cache_dir / f"{idx}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        
        try:
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to max length if specified
            if self.max_audio_len is not None:
                if waveform.shape[1] > self.max_audio_len:
                    waveform = waveform[:, :self.max_audio_len]
                else:
                    padding = self.max_audio_len - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Compute mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax
            )
            mel_spec = mel_transform(waveform)
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {str(e)}")
            raise
        
        try:
            # Get speaker embedding
            speaker_id = row['speaker_id']
            speaker_embedding = self.get_speaker_embedding(speaker_id)
            
            # Get emotion embedding
            emotion = row['emotion']
            emotion_embedding = self.get_emotion_embedding(emotion)
            
            # Convert text to indices
            text_indices = self._text_to_indices(row['text'])
            
            # Create item
            item = {
                'audio': waveform,
                'mel_spec': mel_spec,
                'text': text_indices,
                'speaker_id': speaker_id,
                'emotion': self.emotion_to_idx[emotion],
                'speaker_embedding': speaker_embedding,
                'emotion_embedding': emotion_embedding
            }
            
            # Cache if enabled
            if self.use_cache:
                torch.save(item, cache_path)
            
            return item
            
        except Exception as e:
            logger.error(f"Error creating dataset item for index {idx}: {str(e)}")
            raise
    
    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding"""
        # For now, return a random embedding of the correct input_dim
        input_dim = self.config['model']['speaker_encoder'].get('input_dim', 80)
        return torch.randn(input_dim)
    
    def get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """Get emotion embedding"""
        # For now, return a random embedding
        # In a real implementation, this would load a pre-computed embedding
        embedding_dim = self.config['model']['emotion_encoder'].get('embedding_dim', 64)
        return torch.randn(embedding_dim)

def get_dataloader(
    root_path: str = "data/processed",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    max_audio_length: Optional[int] = None,
    use_cache: bool = True
) -> DataLoader:
    """
    Get dataloader for voice cloning dataset
    
    Args:
        root_path: Path to dataset root (defaults to 'data/processed')
        split: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_audio_length: Maximum audio length in samples
        use_cache: Whether to use cached features
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    root_path = Path(root_path)
    dataset = VoiceCloningDataset(
        metadata_path=root_path / "metadata.csv",
        config_path=Path("config.yaml"),  # Config file in the root directory
        split=split,
        max_audio_len=max_audio_length,
        use_cache=use_cache
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    ) 