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
    n_mels = batch[0]['mel_spec'].size(1)  # Number of mel bands
    
    # Initialize tensors
    batch_size = len(batch)
    mel_specs = torch.zeros(batch_size, max_mel_len, n_mels)
    texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    emotions = torch.zeros(batch_size, dtype=torch.long)
    speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])
    
    # Fill tensors
    for i, item in enumerate(batch):
        mel_len = item['mel_spec'].size(0)
        text_len = item['text'].size(0)
        
        # Ensure mel spectrogram has correct dimensions [time, freq]
        mel_spec = item['mel_spec']
        if len(mel_spec.shape) == 3:  # [batch, time, freq]
            mel_spec = mel_spec.squeeze(0)
        elif len(mel_spec.shape) == 2 and mel_spec.shape[0] != mel_len:  # [freq, time]
            mel_spec = mel_spec.transpose(0, 1)
        
        mel_specs[i, :mel_len] = mel_spec
        texts[i, :text_len] = item['text']
        emotions[i] = item['emotion']
    
    return {
        'mel_spec': mel_specs,
        'text': texts,
        'emotion': emotions,
        'speaker_embedding': speaker_embeddings
    } 