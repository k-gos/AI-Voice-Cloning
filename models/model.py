# Add this at the beginning of your script to disable Flash Attention
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force transformers to not use Flash Attention
import transformers
transformers.utils.is_flash_attn_available = lambda: False

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .text_encoder import TextEncoder
from .speaker_encoder import SpeakerEncoder
from .decoder import MelSpectrogram2Decoder
from .vocoder import HiFiGANVocoder

# Import emotion encoder if it exists
try:
    from .emotion_encoder import EmotionEncoder
    has_emotion_encoder = True
except ImportError:
    has_emotion_encoder = False

"""
Main voice cloning model implementation
"""

class VoiceCloningModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.text_encoder = TextEncoder(config['text_encoder'])
        self.speaker_encoder = SpeakerEncoder(config['speaker_encoder'])
        self.emotion_encoder = EmotionEncoder(config['emotion_encoder'])
        self.mel_proj = nn.Linear(80, config['decoder']['input_dim'])
        self.decoder = MelSpectrogram2Decoder(config['decoder'])
        self.vocoder = HiFiGANVocoder(config['vocoder'])
        
    def forward(
        self,
        text: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_embedding: torch.Tensor,  # This is now emotion indices
        target_mel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Verify input dimensions
        if len(text.shape) != 2:  # [batch, seq_len]
            raise ValueError(f"Expected text of shape [batch, seq_len], got {text.shape}")
        if len(speaker_embedding.shape) != 2:  # [batch, embedding_dim]
            raise ValueError(f"Expected speaker_embedding of shape [batch, embedding_dim], got {speaker_embedding.shape}")
        if len(emotion_embedding.shape) != 1:  # [batch]
            raise ValueError(f"Expected emotion_embedding of shape [batch], got {emotion_embedding.shape}")
        
        # Encode text
        text_features = self.text_encoder(text)  # [batch, seq_len, dim]
        
        # Encode speaker and emotion
        speaker_features = self.speaker_encoder(speaker_embedding)  # [batch, dim]
        emotion_features, emotion_logits = self.emotion_encoder(emotion_embedding)  # [batch, dim]
        
        # Expand speaker and emotion features to match text sequence length
        speaker_features = speaker_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        emotion_features = emotion_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        
        # Combine features
        combined_features = torch.cat([
            text_features,
            speaker_features,
            emotion_features
        ], dim=-1)  # [batch, seq_len, combined_dim]
        
        # Decode mel spectrogram
        if target_mel is not None:
            # Verify target_mel dimensions
            if len(target_mel.shape) != 3:  # [batch, time, freq]
                raise ValueError(f"Expected target_mel of shape [batch, time, freq], got {target_mel.shape}")
            projected_mel = self.mel_proj(target_mel)  # [B, T, 896]
            mel_output = self.decoder(projected_mel, combined_features)
        else:
            # Generate mel spectrogram autoregressively
            mel_output = self._generate_mel(combined_features)
        
        # Generate waveform
        waveform = self.vocoder(mel_output)
        
        return {
            'mel_output': mel_output,  # [batch, time, freq]
            'waveform': waveform,  # [batch, time]
            'emotion_logits': emotion_logits  # [batch, num_emotions]
        }
    
    def _generate_mel(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        max_len = 1000  # Maximum sequence length
        
        # Initialize with zeros [batch, time, freq]
        mel_output = torch.zeros(batch_size, max_len, 80).to(features.device)
        
        # Generate mel spectrogram autoregressively
        for i in range(max_len):
            current_mel = mel_output[:, :i+1, :]
            projected_mel = self.mel_proj(current_mel)
            next_mel = self.decoder(projected_mel, features)
            mel_output[:, i:i+1, :] = next_mel[:, -1:, :]
            
            # Stop if we predict silence
            if torch.all(next_mel[:, -1:, :] < 1e-3):
                break
        
        return mel_output

class VoiceCloningLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_criterion = nn.L1Loss()
        self.stop_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self,
                mel_outputs: torch.Tensor,
                stop_outputs: torch.Tensor,
                mel_targets: torch.Tensor,
                stop_targets: torch.Tensor) -> torch.Tensor:
        mel_loss = self.mel_criterion(mel_outputs, mel_targets)
        stop_loss = self.stop_criterion(stop_outputs, stop_targets)
        return mel_loss + stop_loss