import torch
from typing import List, Dict

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle varying dimensions in the batch
    
    Args:
        batch: List of dataset items
        
    Returns:
        Collated batch with consistent dimensions
    """
    # Get max lengths
    max_mel_len = max(item['mel_spec'].size(0) for item in batch)
    max_text_len = max(item['text'].size(0) for item in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    mel_specs = torch.zeros(batch_size, max_mel_len, batch[0]['mel_spec'].size(1))
    texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    emotions = torch.zeros(batch_size, dtype=torch.long)
    speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])
    
    # Fill tensors
    for i, item in enumerate(batch):
        mel_len = item['mel_spec'].size(0)
        text_len = item['text'].size(0)
        
        mel_specs[i, :mel_len] = item['mel_spec']
        texts[i, :text_len] = item['text']
        emotions[i] = item['emotion']
    
    return {
        'mel_spec': mel_specs,
        'text': texts,
        'emotion': emotions,
        'speaker_embedding': speaker_embeddings
    } 