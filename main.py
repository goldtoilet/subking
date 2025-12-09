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
    st.error("ELEVENLABS_API_KEY 를 설정해주세요.")
    st.stop()

client = ElevenLabs(api_key=ELEVEN_KEY)


# =========================
# 2. 무료 계정에서 사용 가능한 보이스 찾아오기
# =========================
def get_free_voices():
    """무료 보이스만 필터링해서 반환"""
    all_voices = client.voices.get_all().voices
    free_voices = []
    for v in all_voices:
        # 무료 계정에서도 사용 가능한 보이스만 추림
        # 'professional' 태그 없는 보이스만 무료 가능
        if ("professional" not in v.labels):
            free_voices.append(v)
    return free_voices


free_voices = get_free_voices()

if not free_voices:
    st.error("사용 가능한 무료 보이스가 없습니다. ElevenLabs에서 Free Voice 하나 추가해주세요.")
    st.stop()

voice_names = [v.name for v in free_voices]


# =========================
# 3. 대본 → 문장 분리
# =========================
def split_script(text, max_chars=28):
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


def build_subtitles(text, cps=8.0, min_dur=1.5, gap=0.2, max_chars=28):
    seg = split_script(text, max_chars=max_chars)
    subs = []
    now = 0.0
    for idx, s in enumerate(seg, 1):
        dur = max(min_dur, len(s) / cps)
        subs.append({"index": idx, "text": s, "start": now, "end": now + dur})
        now += dur + gap
    return subs


# =========================
# 4. 무료 모델용 ElevenLabs TTS
# =========================
def tts_free(text, voice_id):
    """무료 플랜에서도 항상 돌아가는 TTS"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = tmp.name

    response = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_monolingual_v1",   # 무료 계정 전용 안전 모델
        output_format="mp3_44100_64",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.0,
            use_speaker_boost=False,   # 무료 계정에서 제한될 수 있어 비활성화
        ),
    )

    with open(out_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return out_path


# =========================
# 5. 자막 렌더링
# =========================
def load_font(size):
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


def render_frame(text, w, h, fontsize, bottom, color, bg, ratio):
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
    draw.multiline_text((x, y), wrapped, font=font, fill=color)
    return img


# =========================
# 6. 영상 생성
# =========================
def build_video(audio_path, subtitles, w, h, fontsize, bottom, color, bg, ratio):
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
    out_path = tmp.name

    clip.write_videofile(
        out_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    return out_path


# =========================
# 7. Streamlit UI
# =========================
st.title("🎬 SubKing – ElevenLabs 무료 계정 버전")

script = st.text_area("대본 입력", height=250)

voice_name = st.selectbox("무료 보이스 선택", voice_names)
voice_id = [v.voice_id for v in free_voices if v.name == voice_name][0]

fontsize = st.slider("자막 크기", 30, 80, 60)
bottom = st.slider("아래 여백(px)", 100, 400, 250)
ratio = st.slider("가로폭 비율", 0.5, 0.95, 0.8)
color = st.selectbox("글자색", ["white", "yellow"])
bg = (0, 0, 0)

if st.button("🎥 영상 생성"):
    subs = build_subtitles(script)
    audio_path = tts_free(script, voice_id)
    video_path = build_video(audio_path, subs, 1080, 1920, fontsize, bottom, color, bg, ratio)

    st.success("완료!")
    with open(video_path, "rb") as f:
        st.video(f.read())

    st.download_button("다운로드", data=open(video_path, "rb").read(), file_name="subking_free.mp4")
