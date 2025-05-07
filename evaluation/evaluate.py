#!/usr/bin/env python3
"""
Evaluation script for voice cloning system
"""

import os
import sys
import argparse
import torch
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model import VoiceCloningModel
from src.utils.dataset import VoiceCloningDataset
from src.utils.audio import save_audio
from .metrics import compute_all_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator for voice cloning model"""
    
    def __init__(self, config_path: str, model_path: str, data_dir: str, output_dir: str):
        """
        Initialize evaluator
        
        Args:
            config_path: Path to config file
            model_path: Path to model checkpoint
            data_dir: Path to data directory
            output_dir: Path to save evaluation results
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
        
        # Load model
        self.model = VoiceCloningModel.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup dataset
        self.test_dataset = VoiceCloningDataset(
            metadata_path=self.data_dir / 'processed' / 'metadata.csv',
            config_path=config_path,
            split='test',
            max_audio_len=self.config['dataset']['max_audio_len'],
            use_cache=self.config['dataset']['use_cache']
        )
        
        logger.info(f"Initialized evaluator with {len(self.test_dataset)} test samples")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        all_metrics = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(self.test_dataset)), desc="Evaluating"):
                # Get sample
                sample = self.test_dataset[idx]
                
                # Move to device
                audio = sample['audio'].unsqueeze(0).to(self.device)
                mel_spec = sample['mel_spec'].unsqueeze(0).to(self.device)
                text = sample['text']
                speaker_id = sample['speaker_id']
                emotion = sample['emotion'].unsqueeze(0).to(self.device)
                
                # Get embeddings
                speaker_embedding = self.test_dataset.get_speaker_embedding(speaker_id).unsqueeze(0).to(self.device)
                emotion_embedding = self.test_dataset.get_emotion_embedding(
                    self.config['emotion']['emotions'][emotion.item()]
                ).unsqueeze(0).to(self.device)
                
                # Generate speech
                outputs = self.model(
                    text=text,
                    speaker_embedding=speaker_embedding,
                    emotion_embedding=emotion_embedding,
                    target_mel=mel_spec
                )
                
                # Compute metrics
                metrics = compute_all_metrics(
                    pred_mel=outputs['mel_output'],
                    target_mel=mel_spec,
                    pred_emotions=outputs['emotion_logits'],
                    target_emotions=emotion,
                    pred_audio=outputs['waveform'],
                    target_audio=audio
                )
                
                all_metrics.append(metrics)
                
                # Save sample
                if idx < 10:  # Save first 10 samples
                    save_audio(
                        outputs['waveform'].squeeze(0).cpu(),
                        self.output_dir / f'sample_{idx}_pred.wav',
                        self.config['audio']['sample_rate']
                    )
                    save_audio(
                        audio.squeeze(0).cpu(),
                        self.output_dir / f'sample_{idx}_target.wav',
                        self.config['audio']['sample_rate']
                    )
        
        # Compute average metrics
        avg_metrics = {
            metric: sum(m[metric] for m in all_metrics) / len(all_metrics)
            for metric in all_metrics[0].keys()
        }
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate voice cloning model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(
        config_path=args.config,
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()