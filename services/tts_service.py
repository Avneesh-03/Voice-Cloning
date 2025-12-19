import torch
import os
from TTS.api import TTS

class TTSService:
    def __init__(self, device=None):
        # Default to CPU for laptop stability
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 'YourTTS' is the multi-lingual model used for zero-shot cloning
        self.model_name = "tts_models/multilingual/multi-dataset/your_tts"
        
        # This will download the model (~600MB) on the first run
        self.tts = TTS(self.model_name, progress_bar=True).to(self.device)

    def synthesize(self, text, reference_wav, output_path):
        """
        Takes input text and a reference audio chunk, 
        then generates the cloned voice audio file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_wav,
            language="en",
            file_path=output_path
        )
        return output_path