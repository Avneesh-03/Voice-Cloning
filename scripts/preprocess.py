import librosa
import soundfile as sf
from pathlib import Path
from scripts.utils import check_file_exists, get_path
import os
import numpy as np


TARGET_SR = 16000
MIN_SEGMENT_SEC = 5
MAX_SEGMENT_SEC = 12
SILENCE_TOP_DB = 30


def preprocess_audio(input_filename: str, target_sr: int = 16000):
    """
    Preprocess raw audio and split into chunks for speaker embedding.
    Original function preserved exactly as before.
    """
    input_path = get_path("raw", input_filename)
    check_file_exists(input_path)

    audio, sr = librosa.load(input_path, sr=None, mono=True)
    if len(audio) == 0:
        raise ValueError("Loaded audio is empty.")

    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) == 0:
        raise ValueError("Audio became empty after silence trimming.")

    peak = abs(audio).max()
    if peak > 0:
        audio = audio / peak

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    chunk_duration = 3.0
    chunk_samples = int(chunk_duration * target_sr)
    chunks = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) < chunk_samples * 0.5:
            continue

        chunk_name = f"{input_path.stem}_chunk_{len(chunks)}.wav"
        chunk_path = get_path("preprocessed", chunk_name)
        sf.write(chunk_path, chunk, target_sr)
        chunks.append(chunk_path)

    if not chunks:
        raise RuntimeError("No valid audio chunks were generated.")

    return chunks

def trim_silence(audio, top_db=SILENCE_TOP_DB):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def normalize_volume(audio):
    peak = abs(audio).max()
    if peak > 0:
        return audio / peak
    return audio

def split_audio(audio, sr=TARGET_SR, min_sec=MIN_SEGMENT_SEC, max_sec=MAX_SEGMENT_SEC):
    min_len = min_sec * sr
    max_len = max_sec * sr
    segments = []
    start = 0
    while start < len(audio):
        end = min(start + max_len, len(audio))
        chunk = audio[start:end]
        if len(chunk) >= min_len:
            segments.append(chunk)
        start += max_len
    return segments

def process_long_audio(audio):
    audio = trim_silence(audio)
    audio = normalize_volume(audio)
    segments = split_audio(audio)
    return segments

def preprocess_audio_file(input_filename: str, prefix=""):
    audio, input_path = librosa.load(get_path("raw", input_filename), sr=TARGET_SR, mono=True)
    segments = process_long_audio(audio)
    output_dir = get_path("preprocessed", Path(input_filename).stem)
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for i, seg in enumerate(segments):
        out_path = os.path.join(output_dir, f"{prefix}_seg_{i}.wav")
        sf.write(out_path, seg, TARGET_SR)
        saved_files.append(out_path)
    return saved_files

def preprocess_directory(input_dir="raw", output_dir="preprocessed"):
    input_path = get_path(input_dir)
    output_path = get_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if not file.endswith(".wav"):
            continue
        chunks = preprocess_audio_file(file, prefix=Path(file).stem)
        print(f"Processed {file}, generated {len(chunks)} segments.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Preprocessing Pipeline")
    parser.add_argument("--input_dir", type=str, default="raw")
    parser.add_argument("--output_dir", type=str, default="preprocessed")
    args = parser.parse_args()

    preprocess_directory(args.input_dir, args.output_dir)
    print(" Preprocessing completed successfully")
