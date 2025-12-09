import streamlit as st
from openai import OpenAI
import os
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ColorClip
import numpy as np

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# 1) Text → Speech (TTS)
# -----------------------------
def generate_tts(text, output_path="output.mp3"):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    # 응답은 bytes
    audio_bytes = response.read()
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# -----------------------------
# 2) Whisper → Timestamps
# -----------------------------
def extract_timestamps(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            timestamp_granularities=["word"]  # 또는 ["segment"]
        )
    return transcript


# -----------------------------
# 3) 타임스탬프에서 자막 구간 만들기
# -----------------------------
def build_subtitle_segments(transcript):
    segments = []

    for w in transcript.words:
        segments.append({
            "text": w.word,
            "start": w.start,
            "end": w.end,
        })
    return segments


# -----------------------------
# 4) 영상 생성
# -----------------------------
def create_video(audio_path, segments, output="result.mp4"):

    # Video size
    W, H = 1080, 1920

    clips = []

    # 배경 (검정)
    bg = ColorClip(size=(W, H), color=(0, 0, 0), duration=segments[-1]["end"])
    clips.append(bg)

    # 자막 생성
    for seg in segments:
        txt = seg["text"]
        start = seg["start"]
        end = seg["end"]

        text_clip = TextClip(
            txt,
            font="Arial-Bold",
            fontsize=70,
            color="white",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            align="center",
            size=(W - 200, None),
        ).set_position(("center", H - 300)).set_start(start).set_duration(end - start)

        clips.append(text_clip)

    final = CompositeVideoClip(clips)

    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio)

    final.write_videofile(output, fps=30, codec="libx264", audio_codec="aac")

    return output


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 SubKing - 오디오 + 타임스탬프 자막 자동 생성기")

script = st.text_area("대본을 입력하세요", height=250)

if st.button("🎤 음성 + 자막 영상 생성"):
    if not script.strip():
        st.error("텍스트를 입력하세요!")
        st.stop()

    st.info("TTS 생성 중…")
    audio_path = generate_tts(script)

    st.info("타임스탬프 분석 중…")
    transcript = extract_timestamps(audio_path)

    st.info("자막 구간 생성 중…")
    segments = build_subtitle_segments(transcript)

    st.info("영상 생성 중… 최대 1~2분 정도 소요될 수 있음.")
    video_path = create_video(audio_path, segments)

    st.success("완료!")
    st.video(video_path)
