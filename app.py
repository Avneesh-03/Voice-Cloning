from scripts.preprocess import preprocess_audio
from services.ecapa_service import ECAPASpeakerEncoder
from services.tts_service import TTSService
from services.vocoder_service import VocoderService
import os

def main():
    print("--- Running Voice Cloning MVP Pipeline ---")
    
    # Paths from folder structure
    raw_audio = "data/raw_audio/test.wav"
    preprocessed_dir = "data/preprocessed_audio"
    output_wav = "data/outputs/final_clone.wav"

    # 1. Preprocessing
    print("Executing: Preprocessing")
    chunks = preprocess_audio(raw_audio, preprocessed_dir)

    # 2. ECAPA Embeddings
    print("Executing: ECAPA-TDNN Embedding Extraction")
    encoder = ECAPASpeakerEncoder()
    avg_emb = encoder.encode_folder(preprocessed_dir)

    # 3. TTS Synthesis (YourTTS)
    print("Executing: YourTTS Synthesis")
    tts = TTSService()
    # We use the first chunk as the voice reference
    tts.synthesize("The natural voice cloning system is now fully operational.", chunks[0], output_wav)

    print(f"--- Process Complete. Output saved to: {output_wav} ---")

if __name__ == "__main__":
    main()