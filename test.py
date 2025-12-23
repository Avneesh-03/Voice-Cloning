from services.tts_service import TTSService

def main():
    tts = TTSService()

    output_path = tts.synthesize_audio(
        text="Hello, this is a complete end to end test of my voice cloning system.",
        speaker_wav_path="data/raw_audio/Sample.wav",
        language="en",
        output_filename="test_output.wav"
    )

    print("✅ YourTTS output generated at:", output_path)

if __name__ == "__main__":
    main()
