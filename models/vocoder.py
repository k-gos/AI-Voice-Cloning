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
        return self.vocoder(x) 