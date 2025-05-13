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
    max_mel_len = max(len(item['mel_spec']) for item in batch)
    max_text_len = max(len(item['text']) for item in batch)
    n_mels = batch[0]['mel_spec'].shape[1]  # Number of mel bands
    max_audio_len = max(item['audio'].shape[1] for item in batch)

    # Initialize tensors
    batch_size = len(batch)
    mel_specs = torch.zeros(batch_size, max_mel_len, n_mels)
    texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    emotions = torch.zeros(batch_size, dtype=torch.long)
    speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])
    speaker_ids = [item['speaker_id'] for item in batch]  # Collect speaker IDs
    audios = torch.zeros(batch_size, 1, max_audio_len)
    
    # Fill tensors
    for i, item in enumerate(batch):
        mel_len = len(item['mel_spec'])
        text_len = len(item['text'])
        audio_len = item['audio'].shape[1]
        
        # Ensure mel spectrogram has correct dimensions [time, freq]
        mel_spec = item['mel_spec']
        if len(mel_spec.shape) != 2:
            raise ValueError(f"Expected mel_spec of shape [time, freq], got {mel_spec.shape}")
        if mel_spec.shape[1] != n_mels:
            raise ValueError(f"Expected {n_mels} mel bands, got {mel_spec.shape[1]}")
        
        # Fill mel spectrogram
        mel_specs[i, :mel_len, :] = mel_spec
        
        # Fill text
        texts[i, :text_len] = item['text']
        
        # Fill emotion
        emotions[i] = item['emotion']

        # Fill audio (pad if needed)
        audios[i, 0, :audio_len] = item['audio']

    return {
        'mel_spec': mel_specs,  # [batch, time, freq]
        'text': texts,  # [batch, seq_len]
        'emotion': emotions,  # [batch]
        'speaker_embedding': speaker_embeddings,  # [batch, embedding_dim]
        'speaker_id': speaker_ids,  # [batch]
        'audio': audios  # [batch, 1, audio_len]
    } 