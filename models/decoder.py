import torch
import torch.nn as nn
from typing import Dict

class MelSpectrogram2Decoder(nn.Module):
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