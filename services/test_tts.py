import torch
from services.tts_service import TTSService

def test_model_loading():
    print("--- Starting TTS Service Test ---")
    try:
        # Check if CUDA (GPU) or MPS (Mac GPU) is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        
        print(f"Detecting device: {device}")
        
        # Initialize the service
        service = TTSService(device=device)
        print("✅ Success: TTSService initialized and models loaded.")
        
    except Exception as e:
        print(f"❌ Error: Could not load the service. Details: {e}")

if __name__ == "__main__":
    test_model_loading()