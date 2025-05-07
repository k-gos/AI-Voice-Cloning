import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from src.utils.dataset import VoiceCloningDataset

def get_dataloader(
    metadata_path: str,
    config_path: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    max_audio_length: float = 10.0,
    use_cache: bool = True
) -> DataLoader:
    """
    Get dataloader for voice cloning dataset
    
    Args:
        metadata_path: Path to metadata CSV file
        config_path: Path to config YAML file
        split: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_audio_length: Maximum audio length in seconds
        use_cache: Whether to use cached features
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    dataset = VoiceCloningDataset(
        metadata_path=metadata_path,
        config_path=config_path,
        split=split,
        max_audio_len=max_audio_length,
        use_cache=use_cache
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    ) 