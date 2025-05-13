#!/usr/bin/env python3
"""
Training script for voice cloning system
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.train import Trainer

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
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU training')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Train model
    trainer.train(args.num_epochs)

if __name__ == '__main__':
    main() 