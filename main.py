import os
import re
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

ELEVEN_API_KEY = (
    os.getenv("ELEVENLABS_API_KEY")
    or st.secrets.get("ELEVENLABS_API_KEY", None)
)

if not ELEVEN_API_KEY:
    st.error("ELEVENLABS_API_KEY 환경변수 또는 secrets.toml 에 ELEVENLABS_API_KEY를 설정해 주세요.")
    st.stop()

el_client = ElevenLabs(api_key=ELEVEN_API_KEY)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "NanumGothic.ttf",
        "./NanumGothic.ttf",
        "fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_subtitle_frame(
    text: str,
    video_width: int,
    video_height: int,
    font_size: int,
    bottom_margin: int,
    text_color,
    bg_color,
    max_text_width_ratio: float,
) -> Image.Image:
    img = Image.new("RGB", (video_width, video_height), bg_color)
    if not text.strip():
        return img

    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    max_text_width = int(video_width * max_text_width_ratio)

    words = list(text)
    if not words:
        return img

    lines = []
    current = words[0]
    for ch in words[1:]:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_text_width:
            current = test
        else:
            lines.append(current)
            current = ch
    lines.append(current)
    wrapped = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (video_width - text_w) // 2
    y = video_height - bottom_margin - text_h

    draw.multiline_text((x, y), wrapped, font=font, fill=text_color, align="center")
    return img


def eleven_tts_to_mp3(text: str, voice_id: str, model_id: str, speed: float) -> str:
    response = el_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.7,
            similarity_boost=0.8,
            style=0.2,
            use_speaker_boost=True,
            speed=speed,
        ),
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    with open(tmp_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return tmp_path


def eleven_stt_words_from_audio(audio_path: str, language_code: str = "kor"):
    with open(audio_path, "rb") as f:
        transcription = el_client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            diarize=False,
            tag_audio_events=False,
            language_code=language_code,
        )

    if hasattr(transcription, "words"):
        words = transcription.words
    else:
        words = transcription.get("words", [])
    return words


def build_sentence_segments_from_words(words):
    segments = []
    buf = []
    start_time = None

    def flush_segment(end_time):
        nonlocal buf, start_time, segments
        text = "".join(buf).strip()
        if text:
            segments.append(
                {
                    "index": len(segments) + 1,
                    "start": float(start_time),
                    "end": float(end_time),
                    "text": text,
                }
            )
        buf = []
        start_time = None

    for w in words:
        w_type = w.get("type", "word")
        t = w.get("text", "")
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", w_start))

        if w_type == "audio_event":
            continue

        if t.strip() == "":
            buf.append(" ")
            continue

        if start_time is None:
            start_time = w_start

        buf.append(t)

        if re.search(r"[\.?!。？！…]$", t):
            flush_segment(w_end)

    if buf and words:
        last_end = float(words[-1].get("end", 0.0))
        flush_segment(last_end)

    return segments


def subtitles_to_video(
    audio_path: str,
    segments,
    video_width: int,
    video_height: int,
    font_size: int,
    bottom_margin: int,
    text_color,
    bg_color,
    max_text_width_ratio: float,
    fps: int = 30,
) -> str:
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    def make_frame(t):
        current_text = ""
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                current_text = seg["text"]
                break
        frame_img = draw_subtitle_frame(
            current_text,
            video_width,
            video_height,
            font_size,
            bottom_margin,
            text_color,
            bg_color,
            max_text_width_ratio,
        )
        return np.array(frame_img)

    video_clip = VideoClip(make_frame, duration=duration).set_audio(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp.name
    tmp.close()

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


st.set_page_config(page_title="SubKing – ElevenLabs 자막 싱크", layout="centered")

st.title("🎬 SubKing – ElevenLabs로 음성+자막 영상 만들기")

st.markdown(
    """
1. 대본 전체를 입력하면  
2. ElevenLabs가 음성을 만들고  
3. 같은 오디오를 다시 STT로 분석해서  
4. **단어별 타임스탬프 → 문장 단위 자막**을 만듭니다.

👉 한 번에 **항상 한 문장만 화면에 표시**되도록 처리했어요.
"""
)

script_text = st.text_area(
    "대본 입력",
    height=260,
    placeholder="예) 인간의 마음에는 누구나 빛나는 부분이 숨어 있다. 때로는 그 빛이 세상에 드러나지 못한 채 묻혀 있기도 하다. 우리는 그 숨은 아름다움을 함께 찾아가려 한다...",
)

st.subheader("🎙 ElevenLabs 음성 설정")

voice_id = st.text_input(
    "Voice ID",
    value="pNInz6obpgDQGcFmaJgB",
    help="ElevenLabs 대시보드에서 원하는 목소리의 voice_id를 복사해서 붙여넣으세요.",
)

model_id = st.selectbox(
    "TTS 모델",
    ["eleven_multilingual_v2", "eleven_turbo_v2_5"],
    index=0,
)

voice_speed = st.slider(
    "음성 속도 (1.0 = 기본)",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
)

st.subheader("🎨 자막 스타일")

font_size = st.slider(
    "자막 글자 크기",
    min_value=32,
    max_value=80,
    value=56,
    step=2,
)

bottom_margin = st.slider(
    "화면 아래에서 자막까지 간격 (px)",
    min_value=120,
    max_value=480,
    value=260,
    step=10,
)

max_text_width_ratio = st.slider(
    "자막 가로폭 비율 (화면 대비)",
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

generate_button = st.button("📽 영상 생성", use_container_width=True)

if generate_button:
    if not script_text.strip():
        st.warning("먼저 대본을 입력해 주세요.")
        st.stop()

    if not voice_id.strip():
        st.warning("ElevenLabs Voice ID를 입력해 주세요.")
        st.stop()

    with st.spinner("1/3 ElevenLabs TTS로 음성 생성 중..."):
        audio_path = eleven_tts_to_mp3(
            text=script_text,
            voice_id=voice_id.strip(),
            model_id=model_id,
            speed=voice_speed,
        )

    with st.spinner("2/3 생성된 음성으로 STT 분석 (단어별 타임스탬프 추출) 중..."):
        words = eleven_stt_words_from_audio(audio_path, language_code="kor")

    if not words:
        st.error("STT 결과에서 단어 정보를 가져오지 못했습니다. 대본/언어 설정을 확인해 주세요.")
        st.stop()

    with st.spinner("3/3 단어 타임스탬프를 문장 자막으로 변환 & 영상 렌더링 중..."):
        segments = build_sentence_segments_from_words(words)

        if not segments:
            st.error("단어를 문장 자막으로 변환하지 못했습니다.")
            st.stop()

        lines_preview = []
        for seg in segments[:12]:
            lines_preview.append(
                f"{seg['index']:>2} | {seg['start']:6.2f} → {seg['end']:6.2f} | {seg['text']}"
            )
        st.markdown("### ⏱ 자막 타임라인 (상위 12문장, 항상 한 문장씩)")
        st.code("\n".join(lines_preview), language="text")

        video_path = subtitles_to_video(
            audio_path=audio_path,
            segments=segments,
            video_width=VIDEO_WIDTH,
            video_height=VIDEO_HEIGHT,
            font_size=font_size,
            bottom_margin=bottom_margin,
            text_color=text_color_name,
            bg_color=bg_color,
            max_text_width_ratio=max_text_width_ratio,
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
