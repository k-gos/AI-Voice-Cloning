# AI Voice Cloning System

A deep learning-based voice cloning system with emotion control capabilities.

## Features

- Text-to-speech synthesis with voice cloning
- Emotion-aware speech synthesis
- High-quality audio output
- Support for multiple speakers
- Real-time inference

## Requirements

- Python 3.7+
- PyTorch 1.7.0+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/k-gos/AI-Voice-Cloning.git
cd Speech
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download datasets:

```bash
python data/download_datasets.py --output_dir data/raw
```

2. Preprocess datasets:

```bash
python data/prepare_datasets.py --config config/config.yaml --output_dir data/processed
```

The script will process the following datasets:

- LibriTTS
- VCTK
- Common Voice
- AISHELL-3

## Training

1. Configure training parameters in `config/config.yaml`

2. Start training:

```bash
python script/train.py \
    --config config/config.yaml \
    --data_dir data/processed \
    --output_dir checkpoints \
    --num_epochs 1000
```

## Inference

```python
from models.model import VoiceCloningModel
import torch

# Load model
model = VoiceCloningModel.load_from_checkpoint('checkpoints/best_model.pt')

# Generate speech
text = "Hello, this is a test."
speaker_id = "speaker_1"
emotion = "happy"

waveform = model.generate(
    text=text,
    speaker_id=speaker_id,
    emotion=emotion
)

# Save audio
torchaudio.save('output.wav', waveform, 22050)
```

## Model Architecture

The system consists of several components:

1. Text Encoder: Converts input text to phoneme sequences
2. Speaker Encoder: Extracts speaker characteristics
3. Emotion Encoder: Processes emotion labels
4. Decoder: Generates mel spectrograms
5. Vocoder: Converts mel spectrograms to waveform

## Configuration

Key parameters can be configured in `config/config.yaml`:

- Model architecture
- Training parameters
- Audio processing
- Dataset configuration
- Emotion settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LibriTTS dataset
- VCTK dataset
- Common Voice dataset
- AISHELL-3 dataset
