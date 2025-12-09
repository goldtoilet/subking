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
# ElevenLabs 설정
# =========================
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY", None)
if not ELEVEN_KEY:
    st.error("ELEVENLABS_API_KEY 환경변수 또는 .streamlit/secrets.toml 에 API 키를 설정해주세요.")
    st.stop()

eleven = ElevenLabs(api_key=ELEVEN_KEY)

# 기본 프리셋 목소리 (원하면 나중에 더 추가 가능)
VOICE_PRESETS = {
    "Adam (남, 영어, 저음)": "pNInz6obpgDQGcFmaJgB",
    "Rachel (여, 영어)": "21m00Tcm4TlvDq8ikWAM",
    "Callum (남, 영어)": "N2lVS1w4EtoT3dr4eOWO",
    "Elli (여, 영어)": "MF3mGyEYCl7XYWbV9V6O",
}


# =========================
# 1. 대본 → 자막 문장 리스트
# =========================
def split_script_to_segments(text: str, max_chars_per_sub: int = 28) -> list[str]:
    """
    일반적인 대본을 넣었을 때:
      - 줄바꿈 + 문장부호(., ?, !, 。, ？, ！) 기준으로 한 번 나누고
      - 각 조각이 너무 길면 max_chars_per_sub 길이로 다시 잘라서
        → 한 번에 한 문장(혹은 짧은 구절)만 자막에 나오도록.
    """
    raw_chunks = re.split(r'(?<=[\.!?。？！])\s+|\n+', text.strip())
    segments = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # 너무 긴 문장은 글자 수 기준으로 잘게 나누기
        while len(chunk) > max_chars_per_sub:
            segments.append(chunk[:max_chars_per_sub])
            chunk = chunk[max_chars_per_sub:]
        if chunk:
            segments.append(chunk)

    return segments


def build_subtitles(
    text: str,
    chars_per_second: float = 8.0,
    min_duration: float = 1.5,
    gap_between_lines: float = 0.2,
    max_chars_per_sub: int = 28,
) -> list[dict]:
    segments = split_script_to_segments(text, max_chars_per_sub=max_chars_per_sub)
    subtitles = []
    current_time = 0.0

    for idx, seg in enumerate(segments, start=1):
        seg_len = max(len(seg), 1)
        dur = max(min_duration, seg_len / chars_per_second)
        start = current_time
        end = start + dur
        subtitles.append(
            {"index": idx, "start": start, "end": end, "text": seg}
        )
        current_time = end + gap_between_lines

    return subtitles


# =========================
# 2. TTS (ElevenLabs)
# =========================
def eleven_tts_to_mp3(
    text: str,
    voice_id: str,
    stability: float = 0.6,
    similarity: float = 0.8,
) -> str:
    """
    ElevenLabs Text-to-Speech → mp3 파일 저장
    - model_id: eleven_multilingual_v2 (일반용) 사용
    - VoiceSettings 로 안정성/유사도 조절
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    out_path = Path(tmp.name)

    response = eleven.text_to_speech.convert(
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

    with open(out_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return str(out_path)


# =========================
# 3. PIL 기반 자막 렌더링
# =========================
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text_to_width(draw, text, font, max_width: int) -> str:
    # 한글/영어 섞여 있으니 문자 단위로 줄바꿈
    chars = list(text)
    if not chars:
        return ""

    lines = []
    current = chars[0]
    for ch in chars[1:]:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = ch
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
# 4. 자막 + 음성 → 영상
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
# 5. Streamlit UI
# =========================
st.set_page_config(page_title="SubKing - 대본 → ElevenLabs 음성+자막 영상", layout="centered")

st.title("🎬 SubKing – 대본으로 ElevenLabs 음성+자막 영상 만들기")

st.markdown(
    """
**1단계. 대본 입력**  
- 평소 쓰는 대본처럼 문단으로 쭉 적으면 됩니다.  
- 줄바꿈 / 마침표 기준으로 알아서 자막을 잘라줘요.

**2단계. 자막/화면 + 목소리 선택**  
- 자막 속도, 길이, 위치, 글자 크기 조절  
- ElevenLabs 프리셋 목소리 선택 또는 직접 voice_id 입력

**3단계. 자막 미리보기 → 영상 생성**
"""
)

script_text = st.text_area(
    "대본 입력",
    height=260,
    placeholder="예)\n인류의 기술 발전은 끊임없는 탐색과 도전의 연속이었다. 그 중에서도 자율주행차라는 혁신은 다양한 가능성을 제시하며 현대 사회를 변화시키고 있다.\n특히 FSD 기술은 인간의 개입 없이 차량이 주변 환경을 인식하고 주행을 결정하는 경험을 가능하게 한다.\n...",
)

with st.expander("⏱ 자막 타이밍 / 속도 설정", expanded=True):
    max_chars_per_sub = st.slider(
        "자막 한 줄 최대 글자 수 (긴 문장 자동 분할 기준)",
        min_value=12,
        max_value=40,
        value=28,
        step=2,
    )
    chars_per_second = st.slider(
        "초당 글자 수 (값이 클수록 자막이 빨리 넘어감)",
        min_value=3.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
    )
    min_duration = st.slider(
        "한 문장 최소 표시 시간 (초)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
    gap_between_lines = st.slider(
        "자막 사이 공백 시간 (초)",
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

with st.expander("🎙 ElevenLabs 목소리 설정", expanded=True):
    preset_name = st.selectbox(
        "프리셋 목소리 선택",
        list(VOICE_PRESETS.keys()),
        index=0,
    )
    custom_voice_id = st.text_input(
        "직접 voice_id 사용 (선택, 비워두면 위 프리셋 사용)",
        "",
        placeholder="내가 만든 클론 보이스 ID 등",
    )
    voice_stability = st.slider(
        "Stability (안정성)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )
    voice_similarity = st.slider(
        "Similarity Boost (원래 목소리와의 유사도)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

col1, col2 = st.columns(2)
preview_button = col1.button("🔍 자막 미리보기", use_container_width=True)
generate_button = col2.button("📽 영상 생성", use_container_width=True)

# -------------------------
# 자막 미리보기
# -------------------------
if preview_button:
    if not script_text.strip():
        st.warning("먼저 대본을 입력해주세요.")
    else:
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
            max_chars_per_sub=max_chars_per_sub,
        )

        st.markdown("### 🔍 자막 타임라인 (상위 12개)")
        rows = []
        for sub in subtitles[:12]:
            rows.append(
                f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
            )
        st.code("\n".join(rows) or "자막이 없습니다.", language="text")

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
        st.image(preview_img, caption="자막 화면 미리보기 (1번째 문장 기준)", use_column_width=True)

# -------------------------
# 영상 생성
# -------------------------
if generate_button:
    if not script_text.strip():
        st.warning("먼저 대본을 입력해주세요.")
        st.stop()

    voice_id = custom_voice_id.strip() or VOICE_PRESETS[preset_name]

    with st.spinner("1/3 자막 타임라인 만드는 중..."):
        subtitles = build_subtitles(
            script_text,
            chars_per_second=chars_per_second,
            min_duration=min_duration,
            gap_between_lines=gap_between_lines,
            max_chars_per_sub=max_chars_per_sub,
        )

    with st.spinner("2/3 ElevenLabs로 음성 생성 중..."):
        audio_path = eleven_tts_to_mp3(
            text=script_text,
            voice_id=voice_id,
            stability=voice_stability,
            similarity=voice_similarity,
        )

    st.markdown("### 🔍 자막 타임라인 (상위 12개)")
    rows = []
    for sub in subtitles[:12]:
        rows.append(
            f"{sub['index']:>2} | {sub['start']:6.2f} → {sub['end']:6.2f} | {sub['text']}"
        )
    st.code("\n".join(rows) or "자막이 없습니다.", language="text")

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
    st.image(preview_img, caption="자막 화면 미리보기 (1번째 문장 기준)", use_column_width=True)

    with st.spinner("3/3 영상 렌더링 중... (조금 걸립니다)"):
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
        file_name="subking_elevenlabs.mp4",
        mime="video/mp4",
    )
