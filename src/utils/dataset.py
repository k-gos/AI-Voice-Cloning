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
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
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
        
        # Setup transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax
        )
        
        # Setup cache
        self.use_cache = use_cache
        self.cache = {}
        
        # Setup max audio length
        self.max_audio_len = max_audio_len or self.config['dataset']['max_audio_len']
        
        logger.info(f"Initialized {split} dataset with {len(self.metadata)} samples")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = Path(row['audio_path'])
        if self.use_cache and audio_path in self.cache:
            waveform = self.cache[audio_path]
        else:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Cache if enabled
            if self.use_cache:
                self.cache[audio_path] = waveform
        
        # Trim or pad audio
        if waveform.shape[1] > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        else:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.max_audio_len - waveform.shape[1])
            )
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Get emotion label
        emotion = row['emotion']
        emotion_idx = self.emotion_to_idx[emotion]
        
        return {
            'audio': waveform,
            'mel_spec': mel_spec,
            'text': row['text'],
            'speaker_id': row['speaker_id'],
            'emotion': emotion_idx,
            'duration': row['duration']
        }
    
    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding for a given speaker ID"""
        # This is a placeholder - in a real implementation, you would load
        # pre-computed speaker embeddings or compute them on the fly
        return torch.randn(256)  # 256 is the speaker embedding dimension
    
    def get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """Get emotion embedding for a given emotion"""
        # This is a placeholder - in a real implementation, you would load
        # pre-computed emotion embeddings or compute them on the fly
        return torch.randn(256)  # 256 is the emotion embedding dimension

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