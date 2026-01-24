import torch
from pathlib import Path
from TTS.api import TTS

from scripts.utils import ensure_dir, check_file_exists, get_path


class TTSService:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # YourTTS zero-shot voice cloning model
        self.model_name = "tts_models/multilingual/multi-dataset/your_tts"
        self.tts = TTS(self.model_name).to(self.device)

        # Ensure output directory exists
        ensure_dir("data/outputs")

    def synthesize_audio(
        self,
        text: str,
        speaker_wav_path: str,
        language: str = "en",
        output_filename: str = "yourtts_output.wav"
    ):
        """
        ✅ PRODUCTION-SAFE: Generate final audio using YourTTS.

        Args:
            text: The text to synthesize.
            speaker_wav_path: Path to a preprocessed audio chunk of the speaker.
            language: Language code (default "en").
            output_filename: File name for saving cloned audio.
        """

        if not text.strip():
            raise ValueError("Text cannot be empty")

        speaker_wav_path = Path(speaker_wav_path)
        check_file_exists(speaker_wav_path)  # preprocessed chunk

        output_path = get_path("outputs", output_filename)

        # YourTTS generates audio from the reference preprocessed chunk
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav_path),
            language=language,
            file_path=output_path
        )

        return output_path

    def generate_mel_debug(
        self,
        text: str,
        speaker_wav_path: str,
        language: str = "en"
    ):
        """
        ⚠ DEBUG ONLY — Returns MEL spectrogram for a preprocessed chunk.

        Note:
            - Only used for analysis, not for vocoder synthesis.
        """

        speaker_wav_path = Path(speaker_wav_path)
        check_file_exists(speaker_wav_path)

        outputs = self.tts.synthesizer.tts(
            text=text,
            speaker_wav=str(speaker_wav_path),
            language_name=language,
            return_mel=True
        )

        # outputs is a list of mels (one per sentence)
        mel = outputs[0] if outputs else None
        return mel

from TTS.api import TTS

def load_tts_model():
    
    device = "mps" if torch.backends.mps.available_mps() else "cpu"
    
    
    model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
    return model

def synthesize_speech(text: str, speaker_embedding, style: str = "neutral"):
    """Day 2 & 3: Generate speech with speaker embedding [cite: 70, 71, 111-115]"""
    model = load_tts_model()
    
    # YourTTS generates audio using text and the embedding from the ECAPA service [cite: 4, 113]
    # We use 'en' as the default language for the MVP
    wav = model.tts(text=text, speaker_embedding=speaker_embedding, language="en")
    
    return wav # Returns audio waveform for Yamini's Vocoder service [cite: 116, 121]
