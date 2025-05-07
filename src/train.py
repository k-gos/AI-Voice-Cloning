"""
Training script for voice cloning system
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.model import VoiceCloningModel, VoiceCloningLoss
from models.emotion_encoder import EmotionLoss
from data.dataset_loader import get_dataloader
from src.utils.audio import save_audio
from src.utils.dataset import VoiceCloningDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for voice cloning model"""
    
    def __init__(self, config_path: str, data_dir: str, output_dir: str):
        """
        Initialize trainer
        
        Args:
            config_path: Path to config file
            data_dir: Path to data directory
            output_dir: Path to save checkpoints and logs
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = VoiceCloningModel(self.config['model']).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup loss functions
        self.mel_loss = nn.MSELoss()
        self.emotion_loss = EmotionLoss(self.config['emotion'])
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Setup datasets
        self.train_dataset = VoiceCloningDataset(
            metadata_path=self.data_dir / 'processed' / 'metadata.csv',
            config_path=config_path,
            split='train',
            max_audio_len=self.config['dataset']['max_audio_len'],
            use_cache=self.config['dataset']['use_cache']
        )
        
        self.val_dataset = VoiceCloningDataset(
            metadata_path=self.data_dir / 'processed' / 'metadata.csv',
            config_path=config_path,
            split='val',
            max_audio_len=self.config['dataset']['max_audio_len'],
            use_cache=self.config['dataset']['use_cache']
        )
        
        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Initialized trainer with {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_mel_loss = 0
        total_emotion_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            audio = batch['audio'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            text = batch['text']
            speaker_ids = batch['speaker_id']
            emotion_targets = batch['emotion'].to(self.device)
            
            # Get speaker and emotion embeddings
            speaker_embeddings = torch.stack([
                self.train_dataset.get_speaker_embedding(sid)
                for sid in speaker_ids
            ]).to(self.device)
            
            emotion_embeddings = torch.stack([
                self.train_dataset.get_emotion_embedding(self.config['emotion']['emotions'][e])
                for e in emotion_targets
            ]).to(self.device)
            
            # Forward pass
            outputs = self.model(
                text=text,
                speaker_embedding=speaker_embeddings,
                emotion_embedding=emotion_embeddings,
                target_mel=mel_spec
            )
            
            # Compute losses
            mel_loss = self.mel_loss(outputs['mel_output'], mel_spec)
            emotion_loss = self.emotion_loss(outputs['emotion_logits'], emotion_targets)
            
            # Total loss
            loss = mel_loss + emotion_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            total_mel_loss += mel_loss.item()
            total_emotion_loss += emotion_loss.item()
            
            pbar.set_postfix({
                'mel_loss': mel_loss.item(),
                'emotion_loss': emotion_loss.item()
            })
        
        # Compute average losses
        avg_mel_loss = total_mel_loss / len(self.train_loader)
        avg_emotion_loss = total_emotion_loss / len(self.train_loader)
        
        # Log losses
        self.writer.add_scalar('train/mel_loss', avg_mel_loss, epoch)
        self.writer.add_scalar('train/emotion_loss', avg_emotion_loss, epoch)
        
        return {
            'mel_loss': avg_mel_loss,
            'emotion_loss': avg_emotion_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_mel_loss = 0
        total_emotion_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                audio = batch['audio'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                text = batch['text']
                speaker_ids = batch['speaker_id']
                emotion_targets = batch['emotion'].to(self.device)
                
                # Get speaker and emotion embeddings
                speaker_embeddings = torch.stack([
                    self.val_dataset.get_speaker_embedding(sid)
                    for sid in speaker_ids
                ]).to(self.device)
                
                emotion_embeddings = torch.stack([
                    self.val_dataset.get_emotion_embedding(self.config['emotion']['emotions'][e])
                    for e in emotion_targets
                ]).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    text=text,
                    speaker_embedding=speaker_embeddings,
                    emotion_embedding=emotion_embeddings,
                    target_mel=mel_spec
                )
                
                # Compute losses
                mel_loss = self.mel_loss(outputs['mel_output'], mel_spec)
                emotion_loss = self.emotion_loss(outputs['emotion_logits'], emotion_targets)
                
                # Update totals
                total_mel_loss += mel_loss.item()
                total_emotion_loss += emotion_loss.item()
        
        # Compute average losses
        avg_mel_loss = total_mel_loss / len(self.val_loader)
        avg_emotion_loss = total_emotion_loss / len(self.val_loader)
        
        # Log losses
        self.writer.add_scalar('val/mel_loss', avg_mel_loss, epoch)
        self.writer.add_scalar('val/emotion_loss', avg_emotion_loss, epoch)
        
        return {
            'mel_loss': avg_mel_loss,
            'emotion_loss': avg_emotion_loss
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, num_epochs: int):
        """Train model"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train epoch
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate(epoch)
            
            # Compute total validation loss
            val_loss = val_losses['mel_loss'] + val_losses['emotion_loss']
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}: "
                f"train_mel_loss={train_losses['mel_loss']:.4f}, "
                f"train_emotion_loss={train_losses['emotion_loss']:.4f}, "
                f"val_mel_loss={val_losses['mel_loss']:.4f}, "
                f"val_emotion_loss={val_losses['emotion_loss']:.4f}"
            )

def main():
    parser = argparse.ArgumentParser(description='Train voice cloning model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to save checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config, args.data_dir, args.output_dir)
    
    # Train model
    trainer.train(args.num_epochs)

if __name__ == '__main__':
    main()