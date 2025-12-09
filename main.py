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

# =========================
# OpenAI 클라이언트 설정
# =========================
API_KEY = os.getenv("GPT_API_KEY") or st.secrets.get("GPT_API_KEY", None)

if not API_KEY:
    st.error("GPT_API_KEY 환경변수 또는 .streamlit/secrets.toml 에 API 키를 설정해주세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# =========================
# 자막 관련 유틸
# =========================
def split_text_to_lines(text: str) -> list[str]:
    """
    1) 줄바꿈 기준으로 우선 자막 줄 생성
    2) 만약 줄바꿈이 없다면 문장부호 기준으로 분리
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines

    # fallback: 문장부호 기준
    raw = re.split(r"(?<=[\.!?。？！])\s+", text.strip())
    return [r.strip() for r in raw if r.strip()]


def build_subtitles(
    text: str,
    chars_per_second: float = 8.0,
    min_duration: float = 1.5,
    gap_between_lines: float = 0.2,
) -> list[dict]:
    """
    입력 텍스트 → 자막 리스트
    - 한 줄 = 자막 한 개
    - 각 줄의 길이에 따라 duration 자동 계산
    - duration = max(min_duration, len(line) / chars_per_second)
    - 자막 사이에 gap_between_lines 초 간격
    """
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


# =========================
# TTS (텍스트 → 음성)
# =========================
def generate_tts_audio(text: str) -> str:
    """
    ChatGPT TTS (gpt-4o-mini-tts)로 mp3 파일 생성 후 파일 경로 반환
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = Path(tmp.name)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # 또는 "tts-1", "tts-1-hd"
        voice="alloy",            # 다른 보이스: nova, onyx, coral, ...
        input=text,
        format="mp3",
    ) as response:
        response.stream_to_file(out_path)

    return str(out_path)


# =========================
# 자막 + 음성 → 영상(mp4)
# =========================
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
    """
    - audio_path 의 오디오를 배경으로
    - ColorClip(단색 배경) 위에 TextClip 자막을 타임라인에 맞춰 올려 영상 생성
    """
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


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SubKing - 텍스트 → 자막+음성 영상", layout="centered")

st.title("🎬 SubKing – 텍스트를 자막+음성 영상으로")

st.markdown(
    """
**사용 방법**

- 한 줄이 하나의 자막이 되도록, 줄바꿈해서 텍스트를 입력하면 가장 컨트롤하기 좋아요.
- 왼쪽/아래의 슬라이더로 자막 속도·길이·위치 등을 미세하게 조정한 뒤 **영상 생성**을 눌러주세요.
"""
)

script_text = st.text_area(
    "대본 / 자막 텍스트",
    height=260,
    placeholder="예)\n우리 아빠는 한 번 고장 난 하수 승강장을 여섯 주 동안 퍼 올리는 일을 했어.\n어떤 선생님이 '공부 열심히 해, 안 그러면 저 사람처럼 될 거야'라고 말한 뒤 아이들이 비웃었지.\n...",
)

with st.expander("⏱ 자막 타이밍 / 속도 설정 (미세 조정)", expanded=True):
    chars_per_second = st.slider(
        "초당 글자 수 (값이 커질수록 자막이 빨리 넘어감)",
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
        help="나중에 이미지 배경으로 확장할 수 있어요.",
    )

    if bg_color_name == "black":
        bg_color = (0, 0, 0)
    elif bg_color_name == "dark_gray":
        bg_color = (20, 20, 20)
    else:  # navy_like
        bg_color = (10, 10, 40)

# 세로 영상 해상도 (고정: 1080x1920)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

generate_button = st.button("📽 영상 생성", use_container_width=True)

if generate_button:
    if not script_text.strip():
        st.warning("먼저 대본 텍스트를 입력해주세요.")
        st.stop()

    with st.spinner("1/3 음성 생성 중 (ChatGPT TTS)..."):
        audio_path = generate_tts_audio(script_text)

    with st.spinner("2/3 자막 타임라인 생성 중..."):
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
        )

    # 자막 미리보기
    st.markdown("### 🔍 자막 타임라인 미리보기 (상위 10개)")
    preview_rows = []
    for sub in subtitles[:10]:
        preview_rows.append(
            f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
        )
    st.code("\n".join(preview_rows) or "자막이 없습니다.", language="text")

    with st.spinner("3/3 영상 렌더링 중... (조금 시간이 걸릴 수 있어요)"):
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

    st.success("영상 생성 완료!")

    # 재생 및 다운로드
    with open(video_path, "rb") as vf:
        video_bytes = vf.read()

    st.video(video_bytes)

    st.download_button(
        "💾 영상 다운로드 (mp4)",
        data=video_bytes,
        file_name="subking_output.mp4",
        mime="video/mp4",
    )
