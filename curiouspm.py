import os
import subprocess
import streamlit as st
import requests
from pydub import AudioSegment, silence
import io
import assemblyai as aai
from gtts import gTTS
import re

AZURE_API_KEY = "22ec84421ec24230a3638d1b51e3a7dc"
AZURE_ENDPOINT = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

if os.path.exists("temp_audio.wav"):
    os.remove("temp_audio.wav")

if os.path.exists("generated_audio.wav"):
    os.remove("generated_audio.wav")

if os.path.exists("converted_audio.wav"):
    os.remove("converted_audio.wav")

if os.path.exists("final_video.mp4"):
    os.remove("final_video.mp4")

# Detectin silent segments in audio
def detect_silence(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    silences = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    silences = [((start / 1000), (end / 1000)) for start, end in silences]
    return silences

# Transcribe audio using assemblyAI
def transcribe_audio_streaming(audio_path):

    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} not found.")
        return ""

    aai.settings.api_key = "d57fd048611543d3a34481d97dc72b79"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(audio_path)    
    return transcript.text

# Upload the video file
st.title("🎥 AI Video Audio Correction")
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    video_path = f"temp_video.{video_file.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.video(video_path)

    # Extract audio from video using ffmpeg
    st.write("Extracting audio from the uploaded video...")
    audio_path = "temp_audio.wav"

    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path],
            check=True 
        )

        if os.path.exists(audio_path):
            st.write("Audio extracted successfully.")
            converted_audio_path = "converted_audio.wav"
            subprocess.run(
                ['ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', converted_audio_path],
                check=True
            )
            st.write("Audio converted to mono, 16-bit, 16000 Hz.")
            
            audio_path = converted_audio_path 
        else:
            raise FileNotFoundError(f"{audio_path} not found after ffmpeg execution.")

    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while extracting audio: {e}")
    except FileNotFoundError as e:
        st.error(str(e))

    # Calling the functions
    st.write("Detecting silent or non-speech segments...")
    silences = detect_silence(audio_path)
    
    st.write("Transcribing the audio...")
    transcription = transcribe_audio_streaming(audio_path)

    # Adding silence placeholders in the transcription
    transcription_with_silence = ""
    if transcription:
        for silence_start, silence_end in silences:
            transcription_with_silence += f" [silence from {silence_start}s to {silence_end}s]"


        # Grammar correction with GPT-4
        def correct_transcription_azure_gpt4(transcription, transcription_with_silence):
            headers = {
                "Content-Type": "application/json",
                "api-key": AZURE_API_KEY,
            }
            data = {
                "messages": [{"role": "user", "content" : f"Please correct the following text for grammar and punctuation with no special symbols or no quotes. After each sentence, provide the corresponding timestamp (in seconds) where the sentence starts in the original audio file. Make sure the timestamps are placed inline within the text in required and necessary positions only. Format the response as: <corrected sentence> [<silence from 'start' to 'end'>] Original Text: {transcription} Timestamps: {transcription_with_silence}"}],
                "max_tokens": 500,
                "temperature": 0.7
            }
            response = requests.post(
                f"{AZURE_ENDPOINT}",
                headers=headers,
                json=data
            )

            print("API Response:", response)

            response_data = response.json()
            if 'choices' in response_data:
                return response_data["choices"][0]["message"]["content"].strip()
            elif 'error' in response_data:
                raise ValueError(f"API Error: {response_data['error']['message']}")
            else:
                raise ValueError("Unexpected response format: 'choices' not found.")

        corrected_transcription = correct_transcription_azure_gpt4(transcription, transcription_with_silence)

        # Generating AI voice from corrected transcription using gTTS
        def generate_speech(text, output_audio_file):
            tts = gTTS(text=text, lang='en', slow=False)
            audio_fp = io.BytesIO()
    
            tts.write_to_fp(audio_fp)

            audio_fp.seek(0)
            mp3_audio = AudioSegment.from_file(audio_fp, format="mp3")
            
            wav_fp = io.BytesIO()
            
            mp3_audio.export(wav_fp, format="wav")
            wav_fp.seek(0)

            with open(output_audio_file, "wb") as out:
                out.write(wav_fp.getvalue())


        # Generating AI voice including silence/mute segments
        generated_audio_path = "generated_audio.wav"
        speech_parts = re.split(r'[\[\]]', corrected_transcription)
        speech_parts = [s for s in speech_parts if s.strip()]

        final_audio = AudioSegment.silent(duration=0)
        count = 0

        for part in speech_parts:
            if "silence from" in part:
                count += 1
                if (count > 1):
                    continue

                try:
                    # Handling silence parts
                    part = part.replace("silence from", "")
                    silence_duration = float(part.split("to")[1].split("s")[0].strip()) - float(part.split("to")[0].split("s")[0].strip())
                    silence_segment = AudioSegment.silent(duration=silence_duration * 1000)
                    final_audio += silence_segment
                except:
                    continue
            else:
                count = 0

                # Handling speech parts
                generate_speech(part, generated_audio_path)
                spoken_segment = AudioSegment.from_wav(generated_audio_path)
                final_audio += spoken_segment

        # Speeding up the audio to match the human like audio
        final_audio = final_audio.speedup(playback_speed = 1.21)
        
        final_audio.export(generated_audio_path, format="wav")
        st.audio(generated_audio_path)

        # Replacing the original audio
        output_video_path = "final_video.mp4"
        subprocess.run(f"ffmpeg -i {video_path} -i {generated_audio_path} -c:v copy -map 0:v:0 -map 1:a:0 {output_video_path}", shell=True)
        st.video(output_video_path)

