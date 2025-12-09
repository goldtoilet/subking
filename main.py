import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI
from moviepy.editor import AudioFileClip, VideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =========================
# OpenAI 설정
# =========================
API_KEY = os.getenv("GPT_API_KEY") or st.secrets.get("GPT_API_KEY", None)

if not API_KEY:
    st.error("GPT_API_KEY 환경변수 또는 .streamlit/secrets.toml 에 GPT_API_KEY를 설정해주세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)


# =========================
# 자막 유틸
# =========================
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
            {"index": idx, "start": start, "end": end, "text": line}
        )
        current_time = end + gap_between_lines

    return subtitles


# =========================
# TTS
# =========================
def generate_tts_audio(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = Path(tmp.name)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(out_path)

    return str(out_path)


# =========================
# PIL 기반 자막 렌더링
# =========================
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    try:
        # 파이썬 기본 설치에 자주 포함되는 폰트
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text_to_width(draw, text, font, max_width: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines = []
    current = words[0]
    for w in words[1:]:
        test = current + " " + w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return "\n".join(lines)


def draw_subtitle_frame(
    text: str,
    video_width: int,
    video_height: int,
    subtitle_fontsize: int,
    subtitle_bottom_margin: int,
    text_color: str,
    bg_color,
    max_text_width_ratio: float,
) -> Image.Image:
    img = Image.new("RGB", (video_width, video_height), bg_color)
    if not text.strip():
        return img

    draw = ImageDraw.Draw(img)
    font = _load_font(subtitle_fontsize)

    max_text_width = int(video_width * max_text_width_ratio)
    wrapped = _wrap_text_to_width(draw, text, font, max_text_width)

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (video_width - text_w) // 2
    y = video_height - subtitle_bottom_margin - text_h

    draw.multiline_text((x, y), wrapped, font=font, fill=text_color, align="center")
    return img


# =========================
# 자막 + 음성 → 영상
# =========================
def subtitles_to_video(
    audio_path: str,
    subtitles: list[dict],
    video_width: int,
    video_height: int,
    subtitle_fontsize: int,
    subtitle_bottom_margin: int,
    text_color: str,
    bg_color,
    max_text_width_ratio: float,
    fps: int = 30,
) -> str:
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    def make_frame(t):
        current_text = ""
        for sub in subtitles:
            if sub["start"] <= t < sub["end"]:
                current_text = sub["text"]
                break
        frame_img = draw_subtitle_frame(
            current_text,
            video_width,
            video_height,
            subtitle_fontsize,
            subtitle_bottom_margin,
            text_color,
            bg_color,
            max_text_width_ratio,
        )
        return np.array(frame_img)

    video_clip = VideoClip(make_frame, duration=duration).set_audio(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = tmp.name

    video_clip.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    video_clip.close()
    audio.close()
    return out_path


def generate_preview_image(
    subtitles: list[dict],
    video_width: int,
    video_height: int,
    subtitle_fontsize: int,
    subtitle_bottom_margin: int,
    text_color: str,
    bg_color,
    max_text_width_ratio: float,
) -> Image.Image:
    first_text = subtitles[0]["text"] if subtitles else "미리보기"
    return draw_subtitle_frame(
        first_text,
        video_width,
        video_height,
        subtitle_fontsize,
        subtitle_bottom_margin,
        text_color,
        bg_color,
        max_text_width_ratio,
    )


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SubKing - 텍스트 → 자막+음성 영상", layout="centered")

st.title("🎬 SubKing – 텍스트를 자막+음성 영상으로")

st.markdown(
    """
- 한 줄이 **자막 한 줄**이 되도록 줄바꿈해서 쓰면 가장 컨트롤하기 좋아요.
- 먼저 **자막 미리보기**로 화면 위치/스타일을 보고,
- 그다음 **영상 생성**으로 mp4까지 만들 수 있어요.
"""
)

script_text = st.text_area(
    "대본 / 자막 텍스트",
    height=260,
    placeholder="예)\n우리 아빠는 한 번 고장 난 하수 승강장을 여섯 주 동안 퍼 올리는 일을 했어.\n어떤 선생님이 '공부 열심히 해, 안 그러면 저 사람처럼 될 거야'라고 말한 뒤 아이들이 비웃었지.\n...",
)

with st.expander("⏱ 자막 타이밍 / 속도 설정", expanded=True):
    chars_per_second = st.slider(
        "초당 글자 수 (값이 클수록 자막이 빨리 넘어감)",
        min_value=3.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )
    min_duration = st.slider(
        "한 줄 최소 표시 시간 (초)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
    gap_between_lines = st.slider(
        "자막 사이 간격 (초)",
        min_value=0.0,
        max_value=1.5,
        value=0.2,
        step=0.1,
    )

with st.expander("🎨 자막 스타일 / 화면 설정", expanded=False):
    subtitle_fontsize = st.slider(
        "자막 글자 크기",
        min_value=30,
        max_value=90,
        value=60,
        step=2,
    )
    subtitle_bottom_margin = st.slider(
        "화면 아래에서 자막까지 간격 (px)",
        min_value=100,
        max_value=500,
        value=280,
        step=10,
    )
    max_text_width_ratio = st.slider(
        "자막 가로 폭 비율 (화면 대비)",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
    )

    text_color_name = st.selectbox(
        "자막 색상",
        ["white", "yellow"],
        index=0,
    )

    bg_color_name = st.selectbox(
        "배경 색상",
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

col1, col2 = st.columns(2)
preview_button = col1.button("🔍 자막 미리보기", use_container_width=True)
generate_button = col2.button("📽 영상 생성", use_container_width=True)

if preview_button:
    if not script_text.strip():
        st.warning("먼저 대본 텍스트를 입력해주세요.")
    else:
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
        )

        st.markdown("### 🔍 자막 타임라인 (상위 10개)")
        preview_rows = []
        for sub in subtitles[:10]:
            preview_rows.append(
                f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
            )
        st.code("\n".join(preview_rows) or "자막이 없습니다.", language="text")

        preview_img = generate_preview_image(
            subtitles,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            subtitle_fontsize,
            subtitle_bottom_margin,
            text_color_name,
            bg_color,
            max_text_width_ratio,
        )
        st.image(preview_img, caption="자막 화면 미리보기", use_column_width=True)

if generate_button:
    if not script_text.strip():
        st.warning("먼저 대본 텍스트를 입력해주세요.")
        st.stop()

    with st.spinner("1/3 자막 타임라인 생성 중..."):
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
        )

    with st.spinner("2/3 음성 생성 중 (ChatGPT TTS)..."):
        audio_path = generate_tts_audio(script_text)

    st.markdown("### 🔍 자막 타임라인 (상위 10개)")
    preview_rows = []
    for sub in subtitles[:10]:
        preview_rows.append(
            f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
        )
    st.code("\n".join(preview_rows) or "자막이 없습니다.", language="text")

    preview_img = generate_preview_image(
        subtitles,
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        subtitle_fontsize,
        subtitle_bottom_margin,
        text_color_name,
        bg_color,
        max_text_width_ratio,
    )
    st.image(preview_img, caption="자막 화면 미리보기", use_column_width=True)

    with st.spinner("3/3 영상 렌더링 중... (조금 시간이 걸릴 수 있어요)"):
        video_path = subtitles_to_video(
            audio_path,
            subtitles,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            subtitle_fontsize,
            subtitle_bottom_margin,
            text_color_name,
            bg_color,
            max_text_width_ratio,
            fps=30,
        )

    st.success("영상 생성 완료!")

    with open(video_path, "rb") as vf:
        video_bytes = vf.read()

    st.video(video_bytes)
    st.download_button(
        "💾 영상 다운로드 (mp4)",
        data=video_bytes,
        file_name="subking_output.mp4",
        mime="video/mp4",
    )
