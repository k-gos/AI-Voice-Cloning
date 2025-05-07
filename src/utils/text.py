"""
Text processing utilities for voice cloning system
"""
import re
import torch
import unicodedata
from typing import List, Dict, Optional
import string
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from phonemizer import phonemize

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextProcessor:
    """Text processor for preparing text inputs"""
    
    def __init__(self, use_bert: bool = True, bert_model: str = "bert-base-uncased"):
        """
        Initialize text processor
        
        Args:
            use_bert: Whether to use BERT tokenizer
            bert_model: BERT model name to use
        """
        self.use_bert = use_bert
        
        if use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # Basic punctuation for simple processing
        self.punctuation = '!,.;:?'
        self.whitespace_pattern = re.compile(r'\s+')
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        
        # Replace multiple whitespaces with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> Dict:
        """
        Tokenize text for model input
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokenized output
        """
        if self.use_bert:
            # Use BERT tokenizer
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "tokens": self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            }
        else:
            # Simple tokenization
            words = word_tokenize(text)
            return {"tokens": words}
            
    def add_punctuation(self, text: str, add_period: bool = True) -> str:
        """
        Add punctuation to text if missing
        
        Args:
            text: Input text
            add_period: Whether to add period at end if no punctuation
            
        Returns:
            Text with punctuation
        """
        text = text.strip()
        
        if text and add_period:
            if text[-1] not in self.punctuation:
                text = text + '.'
                
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Use NLTK's sentence tokenizer
        from nltk.tokenize import sent_tokenize
        
        # Make sure text ends with punctuation
        text = self.add_punctuation(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        return sentences


def clean_text(text: str) -> str:
    """
    Clean text for TTS processing
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_to_phonemes(text: str, language: str = 'en-us') -> str:
    """
    Convert text to phonemes
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        Phoneme sequence
    """
    # Clean text
    text = clean_text(text)
    
    # Convert to phonemes
    phonemes = phonemize(
        text,
        language=language,
        backend='espeak',
        strip=True
    )
    
    return phonemes

def prepare_text_for_tts(text: str, max_length: Optional[int] = None) -> List[str]:
    """
    Prepare text for TTS by splitting into chunks
    
    Args:
        text: Input text
        max_length: Maximum chunk length
        
    Returns:
        List of text chunks
    """
    # Clean text
    text = clean_text(text)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Split long sentences if needed
    chunks = []
    for sentence in sentences:
        if max_length and len(sentence) > max_length:
            # Split on punctuation or conjunctions
            sub_chunks = re.split(r'[,;:]|\s+(and|or|but)\s+', sentence)
            sub_chunks = [c.strip() for c in sub_chunks if c.strip()]
            chunks.extend(sub_chunks)
        else:
            chunks.append(sentence)
    
    return chunks

def clean_text_for_filename(text: str) -> str:
    """
    Clean text for use in filenames
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    
    # Remove extra underscores
    text = re.sub(r'_+', '_', text)
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    return text

def text_to_tensor(text: str, vocab: dict) -> torch.Tensor:
    """
    Convert text to tensor
    
    Args:
        text: Input text
        vocab: Vocabulary dictionary
        
    Returns:
        Tensor of token indices
    """
    # Convert to phonemes
    phonemes = text_to_phonemes(text)
    
    # Convert to indices
    indices = [vocab.get(p, vocab['<unk>']) for p in phonemes]
    
    # Add start and end tokens
    indices = [vocab['<sos>']] + indices + [vocab['<eos>']]
    
    return torch.tensor(indices, dtype=torch.long)


if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    test_text = "Hello, world! This is a test. How are you doing today?"
    
    print("Original text:", test_text)
    print("Normalized text:", processor.normalize_text(test_text))
    
    tokenized = processor.tokenize(test_text)
    print("Tokenized:", tokenized["tokens"])
    
    sentences = processor.split_into_sentences(test_text)
    print("Sentences:", sentences)
    
    chunks = prepare_text_for_tts("This is a longer text that should be split into multiple chunks. "
                                 "It contains several sentences of varying length. "
                                 "Some are short. Others are much longer and more complex, "
                                 "containing multiple clauses and phrases.")
    print("TTS chunks:", chunks)