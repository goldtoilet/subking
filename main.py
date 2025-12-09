# subking.py

import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    TextClip,
    CompositeVideoClip,
)

API_KEY = os.getenv("GPT_API_KEY") or st.secrets.get("GPT_API_KEY", None)

if not API_KEY:
    st.error("GPT_API_KEY is missing.")
    st.stop()

client = OpenAI(api_key=API_KEY)


def split_text_to_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    raw = re.split(r"(?<=[\.!?。？！])\s+", text.strip())
    return [r.strip() for r in raw if r.strip()]


def build_subtitles(
    text: str,
    chars_per_second: float = 8.0,
    min_duration: float = 1.5,
    gap_between_lines: float = 0.2,
) -> list[dict]:
    lines = split_text_to_lines(text)
    subtitles = []
    current_time = 0.0

    for idx, line in enumerate(lines, start=1):
        line_len = max(len(line), 1)
        dur = max(min_duration, line_len / chars_per_second)
        start = current_time
        end = start + dur

        subtitles.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "text": line,
            }
        )
        current_time = end + gap_between_lines

    return subtitles


def generate_tts_audio(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = Path(tmp.name)

    # ✅ format 인자 삭제, 모델은 gpt-4o-mini-tts 로 변경
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(out_path)

    return str(out_path)


def subtitles_to_video(
    audio_path: str,
    subtitles: list[dict],
    video_width: int = 1080,
    video_height: int = 1920,
    subtitle_fontsize: int = 60,
    subtitle_bottom_margin: int = 280,
    text_color: str = "white",
    bg_color=(0, 0, 0),
    max_text_width_ratio: float = 0.8,
    fps: int = 30,
) -> str:
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    bg = ColorClip(
        size=(video_width, video_height),
        color=bg_color,
    ).set_duration(duration)

    text_clips = []
    text_width = int(video_width * max_text_width_ratio)

    for sub in subtitles:
        start = sub["start"]
        end = sub["end"]
        line = sub["text"]

        if start >= duration:
            break
        end = min(end, duration)

        txt_clip = (
            TextClip(
                line,
                fontsize=subtitle_fontsize,
                color=text_color,
                method="caption",
                size=(text_width, None),
            )
            .set_start(start)
            .set_end(end)
            .set_position(
                (
                    "center",
                    video_height - subtitle_bottom_margin,
                )
            )
        )

        text_clips.append(txt_clip)

    video = CompositeVideoClip([bg, *text_clips]).set_audio(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = tmp.name

    video.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    video.close()
    audio.close()

    return out_path


st.set_page_config(page_title="SubKing - TTS Subtitle Video", layout="centered")

st.title("🎬 SubKing – Text → Subtitle + Voice Video")

st.markdown(
    """
- Enter text (each line becomes a subtitle)
- Adjust timing and styles
- Generate TTS + subtitles + video
"""
)

script_text = st.text_area(
    "Input Text",
    height=260,
    placeholder="Example:\nLine1...\nLine2...\nLine3...",
)

with st.expander("Timing Controls", expanded=True):
    chars_per_second = st.slider(
        "Characters per second",
        min_value=3.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )
    min_duration = st.slider(
        "Minimum duration per line",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
    gap_between_lines = st.slider(
        "Gap between subtitles",
        min_value=0.0,
        max_value=1.5,
        value=0.2,
        step=0.1,
    )

with st.expander("Style Controls", expanded=False):
    subtitle_fontsize = st.slider(
        "Subtitle font size",
        min_value=30,
        max_value=90,
        value=60,
        step=2,
    )
    subtitle_bottom_margin = st.slider(
        "Bottom margin",
        min_value=100,
        max_value=500,
        value=280,
        step=10,
    )
    max_text_width_ratio = st.slider(
        "Text width ratio",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
    )

    text_color_name = st.selectbox(
        "Text color",
        ["white", "yellow"],
        index=0,
    )

    bg_color_name = st.selectbox(
        "Background color",
        ["black", "dark_gray", "navy_like"],
        index=0,
    )

    if bg_color_name == "black":
        bg_color = (0, 0, 0)
    elif bg_color_name == "dark_gray":
        bg_color = (20, 20, 20)
    else:
        bg_color = (10, 10, 40)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

generate_button = st.button("Generate Video", use_container_width=True)

if generate_button:
    if not script_text.strip():
        st.warning("Enter text first.")
        st.stop()

    with st.spinner("Generating TTS..."):
        audio_path = generate_tts_audio(script_text)

    with st.spinner("Building subtitles..."):
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
        )

    st.markdown("### Preview Subtitles")
    preview_rows = []
    for sub in subtitles[:10]:
        preview_rows.append(
            f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
        )
    st.code("\n".join(preview_rows) or "No subtitles.", language="text")

    with st.spinner("Rendering video..."):
        video_path = subtitles_to_video(
            audio_path,
            subtitles,
            video_width=VIDEO_WIDTH,
            video_height=VIDEO_HEIGHT,
            subtitle_fontsize=subtitle_fontsize,
            subtitle_bottom_margin=subtitle_bottom_margin,
            text_color=text_color_name,
            bg_color=bg_color,
            max_text_width_ratio=max_text_width_ratio,
            fps=30,
        )

    st.success("Video generated!")

    with open(video_path, "rb") as vf:
        video_bytes = vf.read()

    st.video(video_bytes)

    st.download_button(
        "Download Video (mp4)",
        data=video_bytes,
        file_name="subking_output.mp4",
        mime="video/mp4",
    )
