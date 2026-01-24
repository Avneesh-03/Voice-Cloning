import streamlit as st
from pathlib import Path

from scripts.preprocess import preprocess_audio
from services.ecapa_service import ECAPASpeakerEncoder
from services.tts_service import TTSService
from scripts.utils import get_path, ensure_dir

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Voice Cloning MVP", layout="centered")
st.title("🎤 Voice Cloning MVP")

# Upload raw audio
uploaded_file = st.file_uploader("Upload a sample speaker audio (.wav)", type=["wav"])
text_to_speak = st.text_area("Enter text to clone voice for", "Hello! This is a test of voice cloning.")

if st.button("Generate Cloned Audio"):
    if not uploaded_file:
        st.warning("Please upload a speaker audio file first.")
    elif not text_to_speak.strip():
        st.warning("Please enter text to synthesize.")
    else:
        # ------------------------------
        # Save uploaded file to raw_audio folder
        # ------------------------------
        raw_dir = ensure_dir(get_path("raw", ""))
        raw_audio_path = raw_dir / uploaded_file.name
        with open(raw_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded file saved: {raw_audio_path}")

        # ------------------------------
        # Preprocess audio
        # ------------------------------
        st.info("🔹 Preprocessing audio...")
        chunks = preprocess_audio(uploaded_file.name)
        if not chunks:
            st.error("No usable chunks generated from audio.")
        else:
            st.success(f"Preprocessed audio chunks: {len(chunks)}")
            speaker_chunk = chunks[0]  # use first chunk as reference

            # ------------------------------
            # ECAPA embedding (optional, for future use)
            # ------------------------------
            st.info("🔹 Extracting speaker embedding...")
            encoder = ECAPASpeakerEncoder()
            speaker_embedding = encoder.encode_chunks(chunks)
            st.success("✅ Speaker embedding extracted (saved in data/embeddings)")

            # ------------------------------
            # TTS / Voice Cloning using YourTTS
            # ------------------------------
            st.info("🔹 Generating cloned audio...")
            tts = TTSService()
            output_filename = f"cloned_{Path(uploaded_file.name).stem}.wav"
            cloned_audio_path = tts.synthesize_audio(
                text=text_to_speak,
                speaker_wav_path=speaker_chunk,
                output_filename=output_filename
            )
            st.success(f"✅ Cloned audio saved at: {cloned_audio_path}")

            # ------------------------------
            # Play cloned audio in Streamlit
            # ------------------------------
            st.audio(cloned_audio_path)
