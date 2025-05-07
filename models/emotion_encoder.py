"""
Emotion encoder implementation for voice cloning system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class EmotionEncoder(nn.Module):
    """Emotion encoder for voice cloning"""
    
    def __init__(self, config: Dict):
        """
        Initialize emotion encoder
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Setup parameters
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_emotions = len(config['emotions'])
        
        # Create layers
        self.embedding = nn.Embedding(self.num_emotions, self.embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        
        # Emotion classifier
        self.classifier = nn.Linear(self.embedding_dim, self.num_emotions)
    
    def forward(self, emotion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            emotion: Emotion tensor [batch_size]
            
        Returns:
            Tuple of (emotion embedding, emotion logits)
        """
        # Get emotion embedding
        embedding = self.embedding(emotion)
        
        # Encode emotion
        features = self.encoder(embedding)
        
        # Get emotion logits
        logits = self.classifier(features)
        
        return features, logits

class EmotionLoss(nn.Module):
    """Loss function for emotion encoder"""
    
    def __init__(self, config: Dict):
        """
        Initialize emotion loss
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, pred_emotions: torch.Tensor, target_emotions: torch.Tensor) -> torch.Tensor:
        """
        Compute emotion loss
        
        Args:
            pred_emotions: Predicted emotion logits [batch_size, num_emotions]
            target_emotions: Target emotion indices [batch_size]
            
        Returns:
            Emotion loss
        """
        return self.criterion(pred_emotions, target_emotions)