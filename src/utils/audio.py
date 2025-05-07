"""
Core audio processing utilities for voice cloning system
"""

import torch
import torchaudio
import numpy as np
import librosa
from typing import Union, Tuple, Optional
from pathlib import Path

def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 22050,
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Whether to convert to mono
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if necessary
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def save_audio(
    waveform: Union[torch.Tensor, np.ndarray],
    file_path: Union[str, Path],
    sample_rate: int = 22050
) -> None:
    """
    Save audio file
    
    Args:
        waveform: Audio waveform
        file_path: Path to save audio file
        sample_rate: Sample rate
    """
    # Convert numpy array to tensor if necessary
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Ensure waveform is 2D
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Save audio
    torchaudio.save(file_path, waveform, sample_rate)

def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: Optional[int] = 8000
) -> torch.Tensor:
    """
    Compute mel spectrogram
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel spectrogram
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )
    
    return mel_transform(waveform)

def compute_pitch(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 22050,
    fmin: int = 50,
    fmax: int = 500
) -> np.ndarray:
    """
    Compute pitch using YIN algorithm
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Pitch values
    """
    # Convert tensor to numpy if necessary
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Compute pitch
    pitch = librosa.yin(waveform, fmin=fmin, fmax=fmax, sr=sample_rate)
    
    return pitch

def compute_energy(
    waveform: Union[torch.Tensor, np.ndarray],
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute energy
    
    Args:
        waveform: Audio waveform
        frame_length: Frame length
        hop_length: Hop length
        
    Returns:
        Energy values
    """
    # Convert tensor to numpy if necessary
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Compute energy
    energy = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    return energy

def apply_emotion_to_waveform(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    emotion: str = "neutral",
    intensity: float = 1.0
) -> np.ndarray:
    """
    Apply emotion effects to waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        emotion: Target emotion
        intensity: Emotion intensity
        
    Returns:
        Modified waveform
    """
    # Get pitch and energy
    pitch = compute_pitch(waveform, sample_rate)
    energy = compute_energy(waveform)
    
    # Apply emotion-specific modifications
    if emotion == "happy":
        # Increase pitch and energy
        pitch = pitch * (1 + 0.2 * intensity)
        energy = energy * (1 + 0.3 * intensity)
    elif emotion == "sad":
        # Decrease pitch and energy
        pitch = pitch * (1 - 0.2 * intensity)
        energy = energy * (1 - 0.3 * intensity)
    elif emotion == "angry":
        # Increase pitch and energy with more variation
        pitch = pitch * (1 + 0.3 * intensity)
        energy = energy * (1 + 0.4 * intensity)
    elif emotion == "surprised":
        # Sharp increase in pitch and energy
        pitch = pitch * (1 + 0.4 * intensity)
        energy = energy * (1 + 0.5 * intensity)
    elif emotion == "fear":
        # Increase pitch and energy with tremolo
        pitch = pitch * (1 + 0.3 * intensity)
        energy = energy * (1 + 0.4 * intensity)
        # Add tremolo effect
        t = np.arange(len(waveform)) / sample_rate
        tremolo = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
        waveform = waveform * tremolo
    
    # Apply modifications
    modified = librosa.effects.pitch_shift(
        waveform,
        sr=sample_rate,
        n_steps=np.log2(pitch.mean() / pitch[0])
    )
    
    # Apply energy modification
    modified = modified * (energy.mean() / energy[0])
    
    return modified