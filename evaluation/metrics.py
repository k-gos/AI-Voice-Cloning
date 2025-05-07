"""
Evaluation metrics for voice cloning system
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import librosa
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

def compute_mel_loss(pred_mel: torch.Tensor, target_mel: torch.Tensor) -> float:
    """Compute mel spectrogram loss"""
    return F.l1_loss(pred_mel, target_mel).item()

def compute_emotion_accuracy(pred_emotions: torch.Tensor, target_emotions: torch.Tensor) -> float:
    """Compute emotion classification accuracy"""
    pred_labels = torch.argmax(pred_emotions, dim=-1)
    return accuracy_score(target_emotions.cpu(), pred_labels.cpu())

def compute_emotion_f1(pred_emotions: torch.Tensor, target_emotions: torch.Tensor) -> float:
    """Compute emotion classification F1 score"""
    pred_labels = torch.argmax(pred_emotions, dim=-1)
    return f1_score(target_emotions.cpu(), pred_labels.cpu(), average='weighted')

def compute_pitch_correlation(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> float:
    """Compute pitch correlation between predicted and target audio"""
    pred_pitch = librosa.yin(pred_audio.cpu().numpy(), fmin=50, fmax=500)
    target_pitch = librosa.yin(target_audio.cpu().numpy(), fmin=50, fmax=500)
    return pearsonr(pred_pitch, target_pitch)[0]

def compute_energy_correlation(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> float:
    """Compute energy correlation between predicted and target audio"""
    pred_energy = librosa.feature.rms(y=pred_audio.cpu().numpy())[0]
    target_energy = librosa.feature.rms(y=target_audio.cpu().numpy())[0]
    return pearsonr(pred_energy, target_energy)[0]

def compute_all_metrics(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor,
    pred_emotions: torch.Tensor,
    target_emotions: torch.Tensor,
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor
) -> Dict[str, float]:
    """Compute all evaluation metrics"""
    return {
        'mel_loss': compute_mel_loss(pred_mel, target_mel),
        'emotion_accuracy': compute_emotion_accuracy(pred_emotions, target_emotions),
        'emotion_f1': compute_emotion_f1(pred_emotions, target_emotions),
        'pitch_correlation': compute_pitch_correlation(pred_audio, target_audio),
        'energy_correlation': compute_energy_correlation(pred_audio, target_audio)
    } 