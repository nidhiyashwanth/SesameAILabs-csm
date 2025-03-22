# SesameAILabs-csm

A conversational speech model (CSM) implementation by Sesame AI Labs that enables text-to-speech generation with context awareness and consistent audio quality.

## Description

SesameAILabs-csm is a powerful text-to-speech model that can generate natural-sounding speech with context awareness. It supports multiple speakers and maintains consistent audio quality across conversations. The model is fine-tuned to ensure that the audio remains consistent, even in long conversations.

## Features

- Text-to-speech generation with context awareness
- Multi-speaker support
- Natural-sounding speech output
- Contextual conversation handling
- Consistent audio quality across conversations
- Support for custom audio input
- GPU acceleration support

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SesameAILabs/csm.git
cd csm
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Log in to Hugging Face (required for model download):

```python
from huggingface_hub import login
login()
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.4.0
- torchaudio 2.4.0
- transformers 4.49.0
- huggingface_hub 0.28.1
- And other dependencies listed in requirements.txt

## Usage

### Basic Usage

```python
from generator import load_csm_1b
import torchaudio

# Initialize the generator
generator = load_csm_1b(device="cuda")

# Generate speech
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# Save the generated audio
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Contextual Conversation

```python
from generator import load_csm_1b, Segment
import torchaudio

# Initialize the generator
generator = load_csm_1b(device="cuda")

# Define speakers, transcripts, and audio paths
speakers = [0]
transcripts = ["Hey how are you doing."]
audio_paths = ["conversational_b.wav"]

# Function to load and resample audio
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create segments
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

# Generate audio with context
audio = generator.generate(
    text="Your response text here",
    speaker=1,
    context=segments,
    max_audio_length_ms=50_000,
)

# Save the generated audio
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## Model Details

The model is automatically downloaded from the Hugging Face Hub when first used. It includes:

- Encoder model
- Decoder model
- Multiple speaker embeddings
- Configuration files

## Author

Nidhi Yashwanth ([github.com/nidhiyashwanth](https://github.com/nidhiyashwanth))

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sesame AI Labs for developing and maintaining the model
- Hugging Face for hosting the model and providing the transformers library
- The PyTorch team for the deep learning framework
