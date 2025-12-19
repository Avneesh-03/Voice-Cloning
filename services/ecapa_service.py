import torch
import numpy as np
import librosa
import os
from speechbrain.inference.speaker import EncoderClassifier

class ECAPASpeakerEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def encode(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = torch.tensor(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def encode_folder(self, folder_path):
        """Extracts and averages embeddings for all chunks in a folder."""
        embeddings = []
        for file in os.listdir(folder_path):
            if file.endswith(".wav") and file.startswith("chunk_"):
                path = os.path.join(folder_path, file)
                emb = self.encode(path)
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            raise ValueError("No audio chunks found in the specified folder.")
            
        # Average all embeddings to get the unique speaker identity
        speaker_embedding = np.mean(embeddings, axis=0)
        return speaker_embedding