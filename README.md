# Voice Cloning MVP

This project implements a minimal voice cloning system using deep learning. 
A user provides voice samples, and the system generates speech in the same voice 
for any given input text.

The project strictly follows the provided project specification and focuses on 
building a clean, modular, and working MVP.

---

## Features

- **Upload voice samples** (external audio files)
- **Automatic audio preprocessing**:
  - Resampling
  - Mono conversion
  - Silence trimming
  - Normalization
  - Audio splitting
- **Speaker embedding extraction** using ECAPA-TDNN
- **Average speaker embedding** to represent a unique speaker voice
- **Text-to-speech synthesis** using YourTTS
- **Waveform generation** using HiFi-GAN vocoder
- **Save generated audio** for playback or download

---

## Tech Stack

- **Python**
- **PyTorch**
- **SpeechBrain** (ECAPA-TDNN)
- **Hugging Face Transformers** (YourTTS, HiFi-GAN)
- **Librosa**
- **SoundFile**
- **Streamlit** (UI integration planned later)

---

## Project Structure

```text
voice-clone/
├─ app.py
├─ services/
│  ├─ ecapa_service.py
│  ├─ tts_service.py
│  └─ vocoder_service.py
├─ scripts/
│  ├─ preprocess.py
│  └─ utils.py
├─ data/
│  ├─ raw_audio/
│  ├─ preprocessed_audio/
│  ├─ embeddings/
│  └─ outputs/
├─ requirements.txt
└─ README.md
```

## Setup Instructions
### 1. Clone the repository

```
git clone <repository-url>
cd Voice-Cloning
```

### 2. Create and activate virtual environment
```
python -m venv .venv
```
# On Windows:
```
source .venv/Scripts/activate
```
# On macOS/Linux:
```
source .venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
## Current Status
Core voice cloning pipeline is under active development.

UI and deployment will be handled in later stages.

Focus is on correctness, modularity, and MVP completion.

## Notes
GPU acceleration is optional but recommended for faster inference.

CPU-only execution is supported.

Models are loaded once to improve performance.