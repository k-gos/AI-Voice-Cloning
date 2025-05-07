#!/usr/bin/env python3
"""
Script to download datasets for voice cloning
"""

import os
import sys
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def download_libritts(output_dir: Path):
    """Download LibriTTS dataset"""
    logger.info("Downloading LibriTTS dataset...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download train-clean-100
    url = "https://www.openslr.org/resources/60/train-clean-100.tar.gz"
    output_path = output_dir / "train-clean-100.tar.gz"
    
    if not output_path.exists():
        download_file(url, output_path)
        
        # Extract
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall(output_dir)
    
    logger.info("LibriTTS download complete")

def download_vctk(output_dir: Path):
    """Download VCTK dataset"""
    logger.info("Downloading VCTK dataset...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download VCTK
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    output_path = output_dir / "VCTK-Corpus-0.92.zip"
    
    if not output_path.exists():
        download_file(url, output_path)
        
        # Extract
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    logger.info("VCTK download complete")

def download_common_voice(output_dir: Path):
    """Download Common Voice dataset"""
    logger.info("Common Voice dataset requires manual download")
    logger.info("Please visit: https://commonvoice.mozilla.org/en/datasets")
    logger.info("Download the English dataset and place it in: %s", output_dir)

def download_aishell3(output_dir: Path):
    """Download AISHELL-3 dataset"""
    logger.info("AISHELL-3 dataset requires manual download")
    logger.info("Please visit: https://www.aishelltech.com/aishell_3")
    logger.info("Download the dataset and place it in: %s", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Download datasets for voice cloning')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save downloaded datasets')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Download datasets
    download_libritts(output_dir / "libritts")
    download_vctk(output_dir / "vctk")
    download_common_voice(output_dir / "common_voice")
    download_aishell3(output_dir / "aishell3")
    
    logger.info("Dataset download complete")

if __name__ == '__main__':
    main() 