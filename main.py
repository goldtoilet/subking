import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from moviepy.editor import AudioFileClip, VideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


# =========================
# 1. ElevenLabs 설정
# =========================
ELEVEN_KEY = (
    os.getenv("ELEVENLABS_API_KEY")
    or st.secrets.get("ELEVENLABS_API_KEY", None)
)

if not ELEVEN_KEY:
    st.error("❌ ELEVENLABS_API_KEY 를 설정해주세요.")
    st.stop()

eleven = ElevenLabs(api_key=ELEVEN_KEY)

VOICE_PRESETS = {
    "Adam (남)": "pNInz6obpgDQGcFmaJgB",
    "Rachel (여)": "21m00Tcm4TlvDq8ikWAM",
    "Callum (남)": "N2lVS1w4EtoT3dr4eOWO",
    "Elli (여)": "MF3mGyEYCl7XYWbV9V6O",
}


# =========================
# 2. 대본 → 문장(자막) 리스트
# =========================
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
    cps: float = 8.0,
    min_dur: float = 1.5,
    gap: float = 0.2,
    max_chars: int = 28,
):
    seg = split_script(text, max_chars=max_chars)
    subs = []
    now = 0.0

    for idx, s in enumerate(seg, 1):
        dur = max(min_dur, len(s) / cps)
        subs.append({
            "index": idx,
            "text": s,
            "start": now,
            "end": now + dur,
        })
        now += dur + gap

    return subs


# =========================
# 3. ElevenLabs 음성 생성
# =========================
def tts_elevenlabs(
    text: str,
    voice_id: str,
    stability: float,
    similarity: float,
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = Path(tmp.name)

    res = eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # chunk-by-chunk streaming write
    with open(out_path, "wb") as f:
        for chunk in res:
            if chunk:
                f.write(chunk)

    return str(out_path)


# =========================
# 4. 자막 렌더링(PIL)
# =========================
def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    chars = list(text)
    lines = []
    cur = chars[0] if chars else ""

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
    bottom: int,
    color: str,
    bg,
    ratio: float,
):
    img = Image.new("RGB", (w, h), bg)
    if not text.strip():
        return img

    draw = ImageDraw.Draw(img)
    font = load_font(fontsize)
    max_w = int(w * ratio)
    wrapped = wrap_text(draw, text, font, max_w)

    box = draw.multiline_textbbox((0, 0), wrapped, font=font)
    tw, th = box[2], box[3]

    x = (w - tw) // 2
    y = h - bottom - th

    draw.multiline_text((x, y), wrapped, font=font, fill=color, align="center")
    return img


# =========================
# 5. 자막 + 오디오 → 영상 생성
# =========================
def build_video(
    audio_path: str,
    subtitles: list,
    w: int,
    h: int,
    fontsize: int,
    bottom: int,
    color: str,
    bg,
    ratio: float,
    fps: int = 30,
):
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    def frame(t):
        text = ""
        for s in subtitles:
            if s["start"] <= t < s["end"]:
                text = s["text"]
                break
        img = render_frame(text, w, h, fontsize, bottom, color, bg, ratio)
        return np.array(img)

    clip = VideoClip(frame, duration=duration).set_audio(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out = tmp.name

    clip.write_videofile(
        out,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )
    clip.close()
    audio.close()
    return out


# =========================
# 6. Streamlit UI (미리보기 없음)
# =========================
st.set_page_config(page_title="SubKing - ElevenLabs 자막영상", layout="centered")

st.title("🎬 SubKing – ElevenLabs 기반 대본 → 음성·자막 영상 생성기")

script = st.text_area(
    "대본 입력",
    height=280,
    placeholder="대본을 입력하세요...",
)


with st.expander("⏱ 자막 속도/길이 조절"):
    max_chars = st.slider("자막 최대 글자수", 12, 40, 28, 2)
    cps = st.slider("초당 글자 수", 3.0, 20.0, 8.0, 0.5)
    min_dur = st.slider("최소 표시 시간(초)", 0.5, 5.0, 1.5, 0.1)
    gap = st.slider("문장 사이 쉬는 시간(초)", 0.0, 2.0, 0.2, 0.1)

with st.expander("🎨 자막 스타일"):
    fontsize = st.slider("글자 크기", 30, 90, 60, 2)
    bottom = st.slider("아래 여백(px)", 100, 500, 280, 10)
    ratio = st.slider("자막 가로폭 비율", 0.5, 0.95, 0.8, 0.05)
    color = st.selectbox("글자색", ["white", "yellow"])
    bg_choice = st.selectbox("배경색", ["black", "dark_gray", "navy"])
    bg = (0, 0, 0) if bg_choice == "black" else (20, 20, 20) if bg_choice == "dark_gray" else (10, 10, 40)

with st.expander("🎙 ElevenLabs 목소리"):
    preset = st.selectbox("프리셋", list(VOICE_PRESETS.keys()), index=0)
    custom_id = st.text_input("voice_id 직접 입력 (선택)")
    stability = st.slider("Stability", 0.0, 1.0, 0.6, 0.05)
    similarity = st.slider("Similarity Boost", 0.0, 1.0, 0.8, 0.05)

generate = st.button("📽 영상 생성하기", use_container_width=True)

W, H = 1080, 1920

if generate:
    if not script.strip():
        st.warning("대본을 먼저 입력해주세요.")
        st.stop()

    voice_id = custom_id.strip() or VOICE_PRESETS[preset]

    with st.spinner("1/3 자막 생성 중..."):
        subs = build_subtitles(script, cps, min_dur, gap, max_chars)

    with st.spinner("2/3 ElevenLabs 음성 생성 중..."):
        audio_path = tts_elevenlabs(script, voice_id, stability, similarity)

    with st.spinner("3/3 영상 렌더링 중..."):
        video_path = build_video(
            audio_path,
            subs,
            W,
            H,
            fontsize,
            bottom,
            color,
            bg,
            ratio,
        )

    st.success("완료!")
    with open(video_path, "rb") as f:
        st.video(f.read())

    st.download_button(
        "💾 다운로드",
        data=open(video_path, "rb").read(),
        file_name="subking_elevenlabs.mp4",
        mime="video/mp4",
    )
