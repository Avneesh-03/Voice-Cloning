import librosa
import soundfile as sf
from pathlib import Path

from scripts.utils import check_file_exists, get_path


def preprocess_audio(input_filename: str, target_sr: int = 16000):
    """
    Preprocess raw audio and split into chunks for speaker embedding.

    Steps:
    - Load audio
    - Convert to mono
    - Trim silence
    - Normalize
    - Resample
    - Split into chunks
    """

    # Resolve input path safely (ONLY filename allowed)
    input_path = get_path("raw", input_filename)
    check_file_exists(input_path)

    # Load audio (preserve original SR first)
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    if len(audio) == 0:
        raise ValueError("Loaded audio is empty.")

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) == 0:
        raise ValueError("Audio became empty after silence trimming.")

    # Normalize audio
    peak = abs(audio).max()
    if peak > 0:
        audio = audio / peak

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Chunking
    chunk_duration = 3.0  # seconds
    chunk_samples = int(chunk_duration * target_sr)

    chunks = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]

        # Skip very small chunks
        if len(chunk) < chunk_samples * 0.5:
            continue

        chunk_name = f"{input_path.stem}_chunk_{len(chunks)}.wav"
        chunk_path = get_path("preprocessed", chunk_name)

        sf.write(chunk_path, chunk, target_sr)
        chunks.append(chunk_path)

    if not chunks:
        raise RuntimeError("No valid audio chunks were generated.")

    return chunks


# Local test
if __name__ == "__main__":
    chunks = preprocess_audio("sample.wav")
    print("Generated chunks:")
    for c in chunks:
        print(c)
