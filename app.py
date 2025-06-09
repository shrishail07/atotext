import streamlit as st
from faster_whisper import WhisperModel
import tempfile
from pydub import AudioSegment
import json

st.set_page_config(page_title="Audio to English Text", layout="centered")
st.title("🎙 Audio to English Text Transcription")

uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file:
    # Save uploaded MP3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        mp3_path = tmp.name

    # Convert to WAV using pydub
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    st.audio(wav_path, format="audio/wav")

    # Load model
    st.text("Transcribing... please wait.")
    model = WhisperModel("base", compute_type="int8")

    segments, _ = model.transcribe(wav_path, beam_size=5)
    transcript = []
    full_text = ""
    for i, segment in enumerate(segments, start=1):
        speaker = "Speaker" if i % 2 == 0 else "Client"
        entry = {
            "speaker": speaker,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        }
        transcript.append(entry)
        full_text += f"{speaker}: {entry['text']}\n"

    st.subheader("📝 Transcription")
    st.text(full_text)

    json_output = json.dumps(transcript, indent=2)
    st.download_button("📥 Download JSON", json_output, file_name="transcription.json", mime="application/json")
