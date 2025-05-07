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
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load metadata with explicit dtypes
        self.metadata = pd.read_csv(
            metadata_path,
            dtype={
                'text': str,
                'audio_path': str,
                'speaker_id': str,
                'emotion': str,
                'split': str
            }
        )
        
        # Validate metadata
        self._validate_metadata()
        
        # Filter by split
        self.metadata = self.metadata[self.metadata['split'] == split]
        
        # Setup audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_fft = self.config['audio']['n_fft']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio']['fmin']
        self.fmax = self.config['audio']['fmax']
        
        # Setup emotion parameters
        self.emotions = self.config['emotion']['emotions']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        
        # Setup cache
        self.use_cache = use_cache
        self.cache_dir = Path(metadata_path).parent / 'cache' / split
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup max length
        self.max_audio_len = max_audio_len or self.config['dataset']['max_audio_len']
        
        # Setup text processing
        self.vocab_size = self.config['model']['text_encoder']['vocab_size']
        self.max_text_len = 100  # Maximum text sequence length
        
        logger.info(f"Initialized {split} dataset with {len(self.metadata)} samples")
    
    def _validate_metadata(self):
        """Validate metadata format and content"""
        required_columns = ['text', 'audio_path', 'speaker_id', 'emotion', 'split']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in self.metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in metadata: {missing_columns}")
        
        # Check for NaN values
        nan_columns = self.metadata.columns[self.metadata.isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"Found NaN values in columns: {nan_columns}")
        
        # Validate text column
        if not all(isinstance(text, str) for text in self.metadata['text']):
            raise ValueError("Text column contains non-string values")
        
        # Validate audio paths
        invalid_paths = [path for path in self.metadata['audio_path'] if not Path(path).exists()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid audio paths")
        
        # Validate emotions
        invalid_emotions = [emotion for emotion in self.metadata['emotion'] if emotion not in self.emotions]
        if invalid_emotions:
            raise ValueError(f"Found invalid emotions: {set(invalid_emotions)}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
            
        # Simple character-level tokenization
        chars = list(text.lower())
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
        
        # Load and process audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim or pad to max length
        if waveform.shape[1] > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        else:
            padding = self.max_audio_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Compute mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax
        )(waveform)
        
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
            'text': text_indices,  # Now using indices instead of raw text
            'speaker_id': speaker_id,
            'emotion': self.emotion_to_idx[emotion],
            'speaker_embedding': speaker_embedding,
            'emotion_embedding': emotion_embedding
        }
        
        # Cache if enabled
        if self.use_cache:
            torch.save(item, cache_path)
        
        return item
    
    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding"""
        # For now, return a random embedding
        # In a real implementation, this would load a pre-computed embedding
        return torch.randn(self.config['model']['speaker_encoder']['embedding_dim'])
    
    def get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """Get emotion embedding"""
        # For now, return a random embedding
        # In a real implementation, this would load a pre-computed embedding
        return torch.randn(self.config['model']['emotion_encoder']['embedding_dim'])

def get_dataloader(root_path: str,
                  split: str = "train",
                  batch_size: int = 16,
                  num_workers: int = 4,
                  max_audio_length: float = 10.0,
                  use_cache: bool = True) -> DataLoader:
    """
    Get dataloader for voice cloning dataset
    
    Args:
        root_path: Path to dataset root
        split: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_audio_length: Maximum audio length in seconds
        use_cache: Whether to use cached features
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    dataset = VoiceCloningDataset(
        metadata_path=root_path / "metadata.csv",
        config_path=root_path / "config" / "config.yaml",
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