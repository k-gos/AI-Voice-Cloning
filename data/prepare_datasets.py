#!/usr/bin/env python3
"""
Script to prepare datasets for voice cloning
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torchaudio
import librosa
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    """Dataset preparation class"""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize dataset preparator
        
        Args:
            config_path: Path to config file
            output_dir: Directory to save processed datasets
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_fft = self.config['audio']['n_fft']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio']['fmin']
        self.fmax = self.config['audio']['fmax']
        
        # Setup emotion parameters
        self.emotions = self.config['emotion']['emotions']
    
    def prepare_libritts(self, data_dir: Path):
        """Prepare LibriTTS dataset"""
        logger.info("Preparing LibriTTS dataset...")
        
        # Process audio files
        audio_files = []
        for audio_path in tqdm(list(data_dir.rglob("*.wav")), desc="Processing audio"):
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Save processed audio
            output_path = self.output_dir / "audio" / audio_path.relative_to(data_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, waveform, self.sample_rate)
            
            # Get duration
            duration = waveform.shape[1] / self.sample_rate
            
            # Get text from transcript
            transcript_path = audio_path.with_suffix(".txt")
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    text = f.read().strip()
            else:
                text = ""
            
            # Add to list
            audio_files.append({
                'audio_path': str(output_path),
                'text': text,
                'speaker_id': audio_path.parent.name,
                'emotion': 'neutral',  # LibriTTS doesn't have emotion labels
                'duration': duration
            })
        
        return pd.DataFrame(audio_files)
    
    def prepare_vctk(self, data_dir: Path):
        """Prepare VCTK dataset"""
        logger.info("Preparing VCTK dataset...")
        
        # Process audio files
        audio_files = []
        for audio_path in tqdm(list(data_dir.rglob("*.wav")), desc="Processing audio"):
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Save processed audio
            output_path = self.output_dir / "audio" / audio_path.relative_to(data_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, waveform, self.sample_rate)
            
            # Get duration
            duration = waveform.shape[1] / self.sample_rate
            
            # Get text from transcript
            transcript_path = audio_path.with_suffix(".txt")
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    text = f.read().strip()
            else:
                text = ""
            
            # Add to list
            audio_files.append({
                'audio_path': str(output_path),
                'text': text,
                'speaker_id': audio_path.parent.name,
                'emotion': 'neutral',  # VCTK doesn't have emotion labels
                'duration': duration
            })
        
        return pd.DataFrame(audio_files)
    
    def prepare_common_voice(self, data_dir: Path):
        """Prepare Common Voice dataset"""
        logger.info("Preparing Common Voice dataset...")
        
        # Load metadata
        metadata_path = data_dir / "train.tsv"
        if not metadata_path.exists():
            logger.warning("Common Voice metadata not found")
            return pd.DataFrame()
        
        metadata = pd.read_csv(metadata_path, sep='\t')
        
        # Process audio files
        audio_files = []
        for _, row in tqdm(metadata.iterrows(), desc="Processing audio"):
            audio_path = data_dir / "clips" / row['path']
            if not audio_path.exists():
                continue
            
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Save processed audio
            output_path = self.output_dir / "audio" / "common_voice" / audio_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, waveform, self.sample_rate)
            
            # Get duration
            duration = waveform.shape[1] / self.sample_rate
            
            # Add to list
            audio_files.append({
                'audio_path': str(output_path),
                'text': row['sentence'],
                'speaker_id': row['client_id'],
                'emotion': 'neutral',  # Common Voice doesn't have emotion labels
                'duration': duration
            })
        
        return pd.DataFrame(audio_files)
    
    def prepare_aishell3(self, data_dir: Path):
        """Prepare AISHELL-3 dataset"""
        logger.info("Preparing AISHELL-3 dataset...")
        
        # Process audio files
        audio_files = []
        for audio_path in tqdm(list(data_dir.rglob("*.wav")), desc="Processing audio"):
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Save processed audio
            output_path = self.output_dir / "audio" / audio_path.relative_to(data_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, waveform, self.sample_rate)
            
            # Get duration
            duration = waveform.shape[1] / self.sample_rate
            
            # Get text from transcript
            transcript_path = audio_path.with_suffix(".txt")
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    text = f.read().strip()
            else:
                text = ""
            
            # Add to list
            audio_files.append({
                'audio_path': str(output_path),
                'text': text,
                'speaker_id': audio_path.parent.name,
                'emotion': 'neutral',  # AISHELL-3 doesn't have emotion labels
                'duration': duration
            })
        
        return pd.DataFrame(audio_files)
    
    def prepare_datasets(self, data_dir: str):
        """Prepare all datasets"""
        data_dir = Path(data_dir)
        
        # Prepare each dataset
        libritts_df = self.prepare_libritts(data_dir / "libritts")
        vctk_df = self.prepare_vctk(data_dir / "vctk")
        common_voice_df = self.prepare_common_voice(data_dir / "common_voice")
        aishell3_df = self.prepare_aishell3(data_dir / "aishell3")
        
        # Combine datasets
        combined_df = pd.concat([
            libritts_df,
            vctk_df,
            common_voice_df,
            aishell3_df
        ], ignore_index=True)
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(
            combined_df,
            test_size=1 - self.config['dataset']['train_split'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=self.config['dataset']['test_split'] / (self.config['dataset']['val_split'] + self.config['dataset']['test_split']),
            random_state=42
        )
        
        # Add split column
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine back
        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Save metadata
        final_df.to_csv(self.output_dir / "metadata.csv", index=False)
        
        logger.info(f"Dataset preparation complete. Total samples: {len(final_df)}")
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Val samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for voice cloning')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed datasets')
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DatasetPreparator(args.config, args.output_dir)
    
    # Prepare datasets
    preparator.prepare_datasets(args.data_dir)

if __name__ == '__main__':
    main() 