import os
import re
import tempfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont


# =====================================================
# 기본 설정
# =====================================================

ELEVEN_API_KEY = (
    os.getenv("ELEVENLABS_API_KEY")
    or st.secrets.get("ELEVENLABS_API_KEY", None)
)

if not ELEVEN_API_KEY:
    st.error("ELEVENLABS_API_KEY 가 설정되어 있지 않습니다.")
    st.stop()

HEADERS_JSON = {
    "xi-api-key": ELEVEN_API_KEY,
    "Content-Type": "application/json",
}
HEADERS_TTS = {
    "xi-api-key": ELEVEN_API_KEY,
    "Content-Type": "application/json",
    "Accept": "audio/mpeg",
}

VIDEO_W = 1080
VIDEO_H = 1920


# =====================================================
# ElevenLabs REST API
# =====================================================

@st.cache_data(ttl=3600)
def fetch_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    try:
        resp = requests.get(url, headers={"xi-api-key": ELEVEN_API_KEY})
        resp.raise_for_status()
        data = resp.json()
        voices = data.get("voices", data)
        items = []
        for v in voices:
            name = v.get("name", "Unknown")
            vid = v.get("voice_id")
            if not vid:
                continue
            label = name
            items.append({"label": label, "voice_id": vid})
        return items
    except Exception as e:
        st.warning(f"보이스 목록을 불러오지 못했습니다: {e}")
        return []


def eleven_tts_to_mp3(text: str, voice_id: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"

    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": False,
        },
    }

    resp = requests.post(url, headers=HEADERS_TTS, json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS 에러 ({resp.status_code}): {resp.text}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    with open(tmp.name, "wb") as f:
        f.write(resp.content)
    return tmp.name


# =====================================================
# 자막 타이밍 / 렌더링
# =====================================================

def split_script(text: str, max_chars: int = 28) -> list[str]:
    parts = re.split(r'(?<=[\.\!\?。？！])\s+|\n+', text.strip())
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            result.append(p[:max_chars])
            p = p[max_chars:]
        result.append(p)
    return result


def build_subtitles(
    text: str,
    chars_per_second: float = 8.0,
    min_duration: float = 1.5,
    gap: float = 0.2,
    max_chars: int = 28,
):
    segs = split_script(text, max_chars=max_chars)
    subs = []
    t = 0.0
    for idx, s in enumerate(segs, 1):
        dur = max(min_duration, len(s) / max(chars_per_second, 1e-3))
        subs.append({"index": idx, "text": s, "start": t, "end": t + dur})
        t += dur + gap
    return subs


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(draw, text: str, font, max_width: int) -> str:
    chars = list(text)
    if not chars:
        return ""
    lines = []
    cur = chars[0]
    for ch in chars[1:]:
        test = cur + ch
        box = draw.textbbox((0, 0), test, font=font)
        if box[2] <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    lines.append(cur)
    return "\n".join(lines)


def render_frame(
    text: str,
    w: int,
    h: int,
    fontsize: int,
    bottom_margin: int,
    color: str,
    bg_color,
    width_ratio: float,
):
    img = Image.new("RGB", (w, h), bg_color)
    if not text.strip():
        return img
    draw = ImageDraw.Draw(img)
    font = load_font(fontsize)
    max_w = int(w * width_ratio)
    wrapped = wrap_text(draw, text, font, max_w)
    box = draw.multiline_textbbox((0, 0), wrapped, font=font)
    tw, th = box[2], box[3]
    x = (w - tw) // 2
    y = h - bottom_margin - th
    draw.multiline_text((x, y), wrapped, font=font, fill=color, align="center")
    return img


def subtitles_to_video(
    audio_path: str,
    subtitles: list[dict],
    w: int,
    h: int,
    fontsize: int,
    bottom_margin: int,
    color: str,
    bg_color,
    width_ratio: float,
    fps: int = 30,
) -> str:
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    def make_frame(t):
        text = ""
        for s in subtitles:
            if s["start"] <= t < s["end"]:
                text = s["text"]
                break
        img = render_frame(
            text, w, h, fontsize, bottom_margin, color, bg_color, width_ratio
        )
        return np.array(img)

    clip = VideoClip(make_frame, duration=duration).set_audio(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = tmp.name

    clip.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )
    clip.close()
    audio.close()
    return out_path


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="SubKing - ElevenLabs 자막 영상", layout="centered")
st.title("🎬 SubKing – ElevenLabs 기반 대본 → 음성 + 자막 영상")

st.markdown(
    """
1. 대본을 입력합니다.  
2. 자막 속도/스타일과 목소리를 선택합니다.  
3. **영상 생성** 버튼을 누르면 mp4가 만들어집니다.
"""
)

script_text = st.text_area(
    "대본 입력",
    height=260,
    placeholder="예) 인류의 기술 발전은 끊임없는 탐색과 도전의 연속이었다...",
)

with st.expander("⏱ 자막 속도 / 길이 설정", expanded=True):
    max_chars = st.slider("자막 한 덩어리 최대 글자 수", 12, 40, 28, 2)
    cps = st.slider("초당 글자 수 (클수록 빨리 넘어감)", 3.0, 20.0, 8.0, 0.5)
    min_dur = st.slider("문장 최소 표시 시간(초)", 0.5, 5.0, 1.5, 0.1)
    gap = st.slider("문장 사이 공백 시간(초)", 0.0, 2.0, 0.2, 0.1)

with st.expander("🎨 자막 스타일", expanded=False):
    fontsize = st.slider("글자 크기", 30, 90, 60, 2)
    bottom = st.slider("화면 아래 여백(px)", 100, 500, 260, 10)
    width_ratio = st.slider("자막 가로 폭 (화면 비율)", 0.5, 0.95, 0.8, 0.05)
    text_color_name = st.selectbox("글자색", ["white", "yellow"], index=0)
    bg_name = st.selectbox("배경색", ["black", "dark gray", "navy"], index=0)

if bg_name == "black":
    bg_color = (0, 0, 0)
elif bg_name == "dark gray":
    bg_color = (20, 20, 20)
else:
    bg_color = (10, 10, 40)

voices = fetch_voices()
with st.expander("🎙 ElevenLabs 목소리 선택", expanded=True):
    if voices:
        labels = [v["label"] for v in voices]
        idx = st.selectbox("보이스", range(len(labels)), format_func=lambda i: labels[i])
        selected_voice_id = voices[idx]["voice_id"]
    else:
        st.warning("사용 가능한 보이스를 찾지 못했습니다. 기본 Adam 보이스를 사용합니다.")
        selected_voice_id = "pNInz6obpgDQGcFmaJgB"

generate = st.button("📽 영상 생성", use_container_width=True)

if generate:
    if not script_text.strip():
        st.warning("대본을 먼저 입력해주세요.")
        st.stop()

    with st.spinner("1/3 자막 타임라인 생성 중..."):
        subtitles = build_subtitles(
            script_text,
            chars_per_second=cps,
            min_duration=min_dur,
            gap=gap,
            max_chars=max_chars,
        )

    with st.spinner("2/3 ElevenLabs 음성 생성 중..."):
        try:
            audio_path = eleven_tts_to_mp3(script_text, selected_voice_id)
        except Exception as e:
            st.error(f"TTS 생성 중 오류가 발생했습니다: {e}")
            st.stop()

    with st.spinner("3/3 영상 렌더링 중 (조금 걸릴 수 있습니다)..."):
        video_path = subtitles_to_video(
            audio_path,
            subtitles,
            VIDEO_W,
            VIDEO_H,
            fontsize,
            bottom,
            text_color_name,
            bg_color,
            width_ratio,
            fps=30,
        )

    st.success("영상 생성 완료!")

    with open(video_path, "rb") as vf:
        video_bytes = vf.read()

    st.video(video_bytes)
    st.download_button(
        "💾 영상 다운로드 (mp4)",
        data=video_bytes,
        file_name="subking_elevenlabs.mp4",
        mime="video/mp4",
    )
