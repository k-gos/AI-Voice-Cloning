import torch
import torch.nn as nn
from typing import Dict

class SpeakerEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x) 