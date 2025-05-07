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

class TextEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['embedding_dim'],
                nhead=8,
                dim_feedforward=config['hidden_dim'],
                dropout=0.1
            ),
            num_layers=config['n_layers']
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim]
        x = self.encoder(x)
        return x.permute(1, 0, 2)  # [batch, seq_len, dim]

class SpeakerEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class EmotionEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
        self.classifier = nn.Linear(config['output_dim'], config['n_emotions'])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

class Decoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config['input_dim'],
                nhead=8,
                dim_feedforward=config['hidden_dim'],
                dropout=0.1
            ),
            num_layers=config['n_layers']
        )
        self.output = nn.Linear(config['input_dim'], config['output_dim'])
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim]
        memory = memory.permute(1, 0, 2)  # [seq_len, batch, dim]
        x = self.decoder(x, memory)
        x = x.permute(1, 0, 2)  # [batch, seq_len, dim]
        return self.output(x)

class Vocoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.vocoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vocoder(x)

class VoiceCloningModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.text_encoder = TextEncoder(config['text_encoder'])
        self.speaker_encoder = SpeakerEncoder(config['speaker_encoder'])
        self.emotion_encoder = EmotionEncoder(config['emotion_encoder'])
        self.decoder = Decoder(config['decoder'])
        self.vocoder = Vocoder(config['vocoder'])
        
    def forward(
        self,
        text: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_embedding: torch.Tensor,
        target_mel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode text
        text_features = self.text_encoder(text)
        
        # Encode speaker and emotion
        speaker_features = self.speaker_encoder(speaker_embedding)
        emotion_features, emotion_logits = self.emotion_encoder(emotion_embedding)
        
        # Combine features
        combined_features = torch.cat([
            text_features,
            speaker_features.unsqueeze(1).expand(-1, text_features.size(1), -1),
            emotion_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        ], dim=-1)
        
        # Decode mel spectrogram
        if target_mel is not None:
            mel_output = self.decoder(target_mel, combined_features)
        else:
            # Generate mel spectrogram autoregressively
            mel_output = self._generate_mel(combined_features)
        
        # Generate waveform
        waveform = self.vocoder(mel_output)
        
        return {
            'mel_output': mel_output,
            'waveform': waveform,
            'emotion_logits': emotion_logits
        }
    
    def _generate_mel(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        max_len = 1000  # Maximum sequence length
        
        # Initialize with zeros
        mel_output = torch.zeros(batch_size, max_len, 80).to(features.device)
        
        # Generate mel spectrogram autoregressively
        for i in range(max_len):
            current_mel = mel_output[:, :i+1, :]
            next_mel = self.decoder(current_mel, features)
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