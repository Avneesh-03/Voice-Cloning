import torch
import numpy as np
import librosa
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

from scripts.utils import check_file_exists, get_path


class ECAPASpeakerEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def encode(self, audio_path: Path) -> np.ndarray:
        """
        Encode a single preprocessed audio chunk into an embedding.
        """
        audio_path = Path(audio_path)
        check_file_exists(audio_path)

        # Load preprocessed audio (already mono, 16kHz)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)

        return embedding.squeeze(0).cpu().numpy()

    def encode_chunks(
        self,
        chunk_paths: list[Path],
        speaker_id: str = "default_speaker",
        save: bool = True
    ) -> np.ndarray:
        """
        Encode multiple chunks and return averaged speaker embedding.
        """

        if not chunk_paths:
            raise ValueError("No chunk paths provided for embedding extraction.")

        embeddings = []

        for chunk_path in chunk_paths:
            emb = self.encode(chunk_path)
            embeddings.append(emb)

        speaker_embedding = np.mean(np.stack(embeddings), axis=0)

        if save:
            embedding_path = get_path(
                "embeddings",
                f"{speaker_id}_embedding.npy"
            )
            np.save(embedding_path, speaker_embedding)

        return speaker_embedding
