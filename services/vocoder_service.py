# services/vocoder_service.py

import torch
import numpy as np
import soundfile as sf
from speechbrain.pretrained import HIFIGAN

from scripts.utils import ensure_dir, get_path


class VocoderService:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure directories exist
        ensure_dir("data/outputs")
        ensure_dir("data/pretrained/vocoders/hifigan-libritts-16kHz")

        # Load HiFi-GAN vocoder
        self.vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="data/pretrained/vocoders/hifigan-libritts-16kHz",
            run_opts={"device": self.device},
        )

        print("✅ HiFi-GAN vocoder loaded successfully")

    def mel_to_audio(self, mel, output_filename="cloned_output.wav"):
        """
        Convert a MEL spectrogram into waveform and save it.
        Expected MEL shape: [batch, n_mels, frames]
        """

        # ─── Normalize input ──────────────────────────────
        if isinstance(mel, list):
            mel = mel[0]

        if isinstance(mel, (np.ndarray, np.generic)):
            mel = torch.tensor(mel, dtype=torch.float32)

        if not torch.is_tensor(mel):
            raise TypeError(f"Unsupported MEL type: {type(mel)}")

        # ─── Fix MEL shape ────────────────────────────────
        # Case: [frames, n_mels] → transpose
        if mel.ndim == 2 and mel.shape[0] < mel.shape[1]:
            mel = mel.T

        # Add batch dimension → [1, n_mels, frames]
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)

        # Final validation
        if mel.shape[-1] == 0:
            raise ValueError("❌ MEL has zero frames. Check TTS output.")

        print("🔹 MEL shape to vocoder:", mel.shape)

        mel = mel.to(self.device)

        # ─── Vocoder inference ────────────────────────────
        with torch.no_grad():
            wav = self.vocoder(mel)

        wav = wav.squeeze().cpu().numpy()

        # ─── Save output ──────────────────────────────────
        output_path = get_path("outputs", output_filename)
        sf.write(output_path, wav, 22050)

        print(f"✅ Audio saved at: {output_path}")
        return output_path
