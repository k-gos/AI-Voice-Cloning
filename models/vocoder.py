import torch
import torch.nn as nn
from typing import Dict

class HiFiGANVocoder(nn.Module):
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
        # Handle input dimensions
        if len(x.shape) == 4:  # [batch, channels, time, freq]
            x = x.squeeze(1)  # Remove channel dimension
        elif len(x.shape) != 3:  # Should be [batch, time, freq]
            raise ValueError(f"Expected input tensor of shape [batch, time, freq] or [batch, channels, time, freq], got {x.shape}")
            
        # Process through vocoder
        return self.vocoder(x) 