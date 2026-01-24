import sys
import os
import torch

# This line tells Python to look one folder up so it can find 'services'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tts_service import TTSService

def run_cloning():
    # Detect Mac GPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")
    
    # 1. Initialize the service
    service = TTSService(device=device)
    
    # 2. Setup paths
    # Note: This assumes your voice sample is in the main Voice-Cloning folder
    reference_wav = "../input_voice.wav" 
    output_path = "../cloned_output.wav"
    text_to_speak = "Testing the system from inside the services folder. It works!"

    print(f"--- Generating audio... ---")
    
    # 3. Generate
    service.generate_voice(
        text=text_to_speak,
        speaker_wav=reference_wav,
        language="en",
        output_path=output_path
    )
    
    print(f"✅ Success! Cloned audio saved to: {output_path}")

if __name__ == "__main__":
    run_cloning()