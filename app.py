 import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
import json

st.set_page_config(page_title="Audio to English Text", layout="centered")
st.title("🎙 Audio to English Text Transcription")

uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        mp3_path = tmp.name

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")
    st.text("Transcribing... Please wait.")

    model = whisper.load_model("base")
    result = model.transcribe(wav_path)
    full_text = result["text"]

    st.subheader("📝 Transcription")
    st.text_area("Output", full_text, height=300)

    json_output = json.dumps(result, indent=2)
    st.download_button("📥 Download JSON", json_output, file_name="transcription.json", mime="application/json")
