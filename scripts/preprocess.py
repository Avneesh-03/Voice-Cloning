import os
import librosa
import soundfile as sf


def preprocess_audio(input_path, output_dir, target_sr=16000):
    """
    Basic audio preprocessing:
    - Load audio
    - Convert to mono
    - Resample to target sample rate
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load audio (keep original sample rate first)
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Remove leading and trailing silence
    
    audio, _ = librosa.effects.trim(audio, top_db=25)


    # Normalize audio to [-1, 1]
    peak = max(abs(audio))
    if peak > 0:
        audio = audio / peak


    # Resample if required
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    output_path = os.path.join(output_dir, "processed.wav")

    # Save processed audio
    sf.write(output_path, audio, target_sr)

    return output_path


#It scales the audio signal so all samples have consistent loudness, which improves embedding quality and TTS stability.

#We’ll do peak normalization (simple, safe, MVP-friendly).