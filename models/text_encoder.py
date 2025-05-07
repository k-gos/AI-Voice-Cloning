import torch
import torch.nn as nn
from typing import Dict

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