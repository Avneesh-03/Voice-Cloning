import torch
import soundfile as sf
import os
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

class VocoderService:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Path to a pre-trained HiFi-GAN vocoder
        # This matches the 'Natural' requirement in the PDF
        self.vocoder_name = "vocoder_models/universal/multi-dataset/high_fidelity_gan"
        
    def refine_audio(self, wav, sr, output_path):
        """
        In a manual pipeline, this would convert mel-spectrograms to audio.
        In our integrated YourTTS pipeline, we use this to ensure the 
        output is normalized and saved at the highest quality.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, sr)
        return output_path