import os
from typing import Optional

import streamlit as st
from openai import OpenAI

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
    ImageClip,
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

# ====================================
# 페이지 설정 (사이드바 항상 펼쳐두기!!)
# ====================================
st.set_page_config(
    page_title="SubKing",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# OpenAI 클라이언트 설정
# =========================
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY 가 설정되어 있지 않습니다.\n\n"
        "- Streamlit Cloud의 'Edit secrets'에서\n"
        '  OPENAI_API_KEY = "sk-..." 형식으로 추가해 주세요.'
    )
    st.stop()

client = OpenAI(api_key=api_key)

# 폰트 (레포 루트에 NanumGothic.ttf 파일이 있다고 가정)
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")


# ====================================
# 0) Pillow로 텍스트 이미지를 만드는 함수
# ====================================
def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """항상 나눔고딕을 우선 사용 (없으면 기본 폰트)."""
    if os.path.isfile(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            pass
    # 폴백
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def hex_to_rgb(color_hex: str):
    """#RRGGBB 형태를 (R,G,B) 튜플로 변환."""
    try:
        return ImageColor.getrgb(color_hex)
    except Exception:
        return (255, 255, 255)


def make_text_image(
    text: str,
    width: int,
    font_size: int,
    text_color_hex: str,
    outline_color_hex: str,
    outline_width: int,
    line_spacing: int = 8,
    align: str = "center",  # "left", "center", "right"
):
    """
    Pillow를 이용해 텍스트 이미지를 생성.
    폭(width)에 맞게 자동 줄바꿈.
    align 파라미터로 좌/중앙/우 정렬 가능.
    """
    if not text:
        text = " "

    font = load_font(font_size)
    text_color = hex_to_rgb(text_color_hex)
    outline_color = hex_to_rgb(outline_color_hex)

    # 줄바꿈 계산용 더미 이미지
    dummy_img = Image.new("RGBA", (width, font_size * 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    words = text.split(" ")
    lines = []
    current_line = ""
    for w in words:
        trial = (current_line + " " + w).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= width:
            current_line = trial
        else:
            if current_line:
                lines.append(current_line)
            current_line = w
    if current_line:
        lines.append(current_line)

    line_height = font_size + line_spacing
    img_height = line_height * len(lines)

    img = Image.new("RGBA", (width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]

        if align == "left":
            x = 0
        elif align == "right":
            x = width - line_width
        else:
            x = (width - line_width) // 2

        # 외곽선
        if outline_width > 0:
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=outline_color)

        # 본 텍스트
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height

    return img


# ====================================
# 1) 텍스트 -> 음성 (OpenAI TTS)
# ====================================
def generate_tts(
    text: str,
    voice: str = "alloy",
    output_path: str = "tts_audio.mp3",
) -> str:
    """
    텍스트를 OpenAI TTS로 mp3 파일로 저장.
    voice 파라미터로 목소리 선택.
    """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    )

    audio_bytes = response.read()

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


# ====================================
# 2) 음성 -> 타임스탬프 (Whisper)
# ====================================
def extract_word_timestamps(audio_path: str):
    """
    Whisper(whisper-1)로 단어 단위 타임스탬프 추출.
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = getattr(transcript, "words", None)
    if words is None and isinstance(transcript, dict):
        words = transcript.get("words", [])

    if words is None:
        words = []

    return words


# ====================================
# 3-A) 단어 리스트를 더 긴 자막 덩어리로 그룹핑
# ====================================
def normalize_words(words):
    """Whisper 결과를 dict 리스트로 정규화."""
    norm = []
    for w in words:
        if hasattr(w, "word"):
            norm.append({"word": w.word, "start": w.start, "end": w.end})
        else:
            norm.append(
                {"word": w["word"], "start": w["start"], "end": w["end"]}
            )
    return norm


def group_words_to_chunks(
    words,
    min_duration: float = 1.2,  # 최소 자막 유지 시간(초)
    max_chars: int = 25,        # 한 자막 블록의 최대 글자 수
):
    """
    너무 자주 바뀌지 않도록 단어들을 묶어서 한 블록으로 만드는 함수.
    """
    words = normalize_words(words)
    chunks = []
    current_text = ""
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for w in words:
        word = w["word"]
        start = w["start"]
        end = w["end"]

        if current_text == "":
            current_text = word
            current_start = start
            current_end = end
        else:
            trial = current_text + " " + word
            trial_len = len(trial)
            duration = end - (current_start if current_start is not None else start)

            if duration >= min_duration or trial_len > max_chars:
                chunks.append(
                    {
                        "text": current_text,
                        "start": current_start,
                        "end": current_end,
                    }
                )
                current_text = word
                current_start = start
                current_end = end
            else:
                current_text = trial
                current_end = end

    if current_text and current_start is not None and current_end is not None:
        chunks.append(
            {"text": current_text, "start": current_start, "end": current_end}
        )

    return chunks


# ====================================
# 3-B) 타임스탬프 기반 자막 + 배경 클립 생성
# ====================================
def build_video_clips_from_chunks(
    chunks,
    video_size=(1080, 1920),
    font_size: int = 70,
    text_color_hex: str = "#FFFFFF",
    outline_color_hex: str = "#000000",
    outline_width: int = 3,
    y_ratio: float = 0.8,  # 0.0(맨 위) ~ 1.0(맨 아래)
    line_spacing: int = 8,
):
    """
    자막 블록(chunks) 리스트로부터 자막 이미지 클립 + 배경 클립 생성.
    """
    W, H = video_size
    clips = []

    if not chunks:
        return clips, 0.0

    last_end = max(c["end"] for c in chunks)

    bg = ColorClip(size=(W, H), color=(0, 0, 0), duration=last_end)
    clips.append(bg)

    y_pos = int(H * y_ratio)

    for c in chunks:
        txt = c["text"]
        start = c["start"]
        end = c["end"]
        if end <= start:
            continue
        duration = end - start

        img = make_text_image(
            txt,
            width=W - 200,
            font_size=font_size,
            text_color_hex=text_color_hex,
            outline_color_hex=outline_color_hex,
            outline_width=outline_width,
            line_spacing=line_spacing,
            align="center",
        )

        img_array = np.array(img)
        text_clip = (
            ImageClip(img_array)
            .set_duration(duration)
            .set_start(start)
            .set_position(("center", y_pos))
        )

        clips.append(text_clip)

    return clips, last_end


# ====================================
# 3-C) 제목(4줄) 클립 생성
# ====================================
def build_title_clips(
    title_lines,
    video_size,
    duration,
    font_size: int,
    outline_width: int,
    line_spacing: int,
    text_colors,
    outline_colors,
    aligns,
    top_ratio: float,
):
    """
    제목 4줄을 화면 상단부터 세로로 배치하는 클립 생성.
    각 줄은 색상/외곽선색/정렬을 개별 설정.
    """
    W, H = video_size
    clips = []
    safe_width = W - 200  # 좌우 여백 확보

    y = int(H * top_ratio)

    for idx, line in enumerate(title_lines):
        if not line or not line.strip():
            continue

        text_color = text_colors[idx] if text_colors and idx < len(text_colors) else "#FFFFFF"
        outline_color = outline_colors[idx] if outline_colors and idx < len(outline_colors) else "#000000"
        align = aligns[idx] if aligns and idx < len(aligns) else "center"

        img = make_text_image(
            line,
            width=safe_width,
            font_size=font_size,
            text_color_hex=text_color,
            outline_color_hex=outline_color,
            outline_width=outline_width,
            line_spacing=line_spacing,
            align=align,
        )

        img_array = np.array(img)
        clip = (
            ImageClip(img_array)
            .set_duration(duration)
            .set_start(0)  # 전체 구간 동안 노출
            .set_position(("center", y))
        )
        clips.append(clip)

        y += font_size + line_spacing

    return clips


# ====================================
# 4) 음성 + 자막(+제목) -> mp4 영상 만들기
# ====================================
def create_video_with_subtitles(
    audio_path: str,
    words,
    video_size=(1080, 1920),
    font_size: int = 70,
    text_color_hex: str = "#FFFFFF",
    outline_color_hex: str = "#000000",
    outline_width: int = 3,
    y_ratio: float = 0.8,
    output_path: str = "subking_result.mp4",
    # --- 제목 관련 옵션 ---
    title_lines=None,
    title_aligns=None,
    title_text_colors=None,
    title_outline_colors=None,
    title_font_size: int = 80,
    title_outline_width: int = 4,
    title_line_spacing: int = 10,
    title_top_ratio: float = 0.1,
):
    if title_lines is None:
        title_lines = []
    if title_aligns is None:
        title_aligns = []
    if title_text_colors is None:
        title_text_colors = []
    if title_outline_colors is None:
        title_outline_colors = []

    chunks = group_words_to_chunks(words)
    clips, duration = build_video_clips_from_chunks(
        chunks,
        video_size=video_size,
        font_size=font_size,
        text_color_hex=text_color_hex,
        outline_color_hex=outline_color_hex,
        outline_width=outline_width,
        y_ratio=y_ratio,
        line_spacing=8,
    )
    if duration <= 0:
        return None

    # 제목 클립 추가
    if any(line.strip() for line in title_lines):
        title_clips = build_title_clips(
            title_lines=title_lines,
            video_size=video_size,
            duration=duration,
            font_size=title_font_size,
            outline_width=title_outline_width,
            line_spacing=title_line_spacing,
            text_colors=title_text_colors,
            outline_colors=title_outline_colors,
            aligns=title_aligns,
            top_ratio=title_top_ratio,
        )
        clips.extend(title_clips)

    video = CompositeVideoClip(clips)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)

    video.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    return output_path


# ====================================
# 5) 미리보기 이미지 생성 (Streamlit UI용)
# ====================================
def create_preview_frame(
    video_size=(1080, 1920),
    font_size: int = 70,
    text_color_hex: str = "#FFFFFF",
    outline_color_hex: str = "#000000",
    outline_width: int = 3,
    y_ratio: float = 0.8,
    sample_text: str = "여기서는 자막이 올라갑니다",
):
    W, H = video_size

    bg = Image.new("RGB", (W, H), (0, 0, 0))

    subtitle_img = make_text_image(
        sample_text,
        width=W - 200,
        font_size=font_size,
        text_color_hex=text_color_hex,
        outline_color_hex=outline_color_hex,
        outline_width=outline_width,
        line_spacing=8,
        align="center",
    )

    sw, sh = subtitle_img.size
    y_pos = int(H * y_ratio) - sh // 2
    x_pos = (W - sw) // 2

    bg.paste(subtitle_img, (x_pos, y_pos), subtitle_img)

    preview = bg.resize((W // 2, H // 2), Image.LANCZOS)
    return preview


# ====================================
# 6) Streamlit UI
# ====================================

# ---------- 왼쪽 사이드바 ----------
side = st.sidebar
side.title("⚙️ SubKing 설정")

# 영상 비율 선택
ratio_label = side.radio(
    "영상 비율 선택",
    ("9:16 쇼츠 (1080x1920)", "16:9 롤폼 (1920x1080)"),
)

if "9:16" in ratio_label:
    video_size = (1080, 1920)
else:
    video_size = (1920, 1080)

side.markdown("---")

# TTS 목소리 선택
voice_options = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]
selected_voice = side.selectbox("🎙 TTS 목소리 선택", options=voice_options, index=0)

side.markdown("---")

# 자막 스타일 (Disclosure / Expander)
with side.expander("🎨 자막 스타일", expanded=True):
    font_size = st.slider(
        "자막 폰트 크기", min_value=40, max_value=120, value=80, step=2
    )
    text_color = st.color_picker("자막 글자 색상", "#FFFFFF")

    outline_width = st.slider(
        "텍스트 외곽선 두께", min_value=0, max_value=8, value=4
    )
    outline_color = st.color_picker("외곽선 색상", "#000000")

    pos_percent = st.slider(
        "자막 세로 위치 (0 = 맨 위, 100 = 맨 아래)",
        min_value=50,
        max_value=95,
        value=80,
    )
    y_ratio = pos_percent / 100.0

    st.markdown("---")
    st.subheader("👀 자막 미리보기")
    preview_img = create_preview_frame(
        video_size=video_size,
        font_size=font_size,
        text_color_hex=text_color,
        outline_color_hex=outline_color,
        outline_width=outline_width,
        y_ratio=y_ratio,
        sample_text="여기서는 자막이 올라갑니다",
    )
    st.image(preview_img, use_container_width=True, caption="현재 자막 설정 미리보기")

# 제목 스타일 (Disclosure / Expander)
with side.expander("📝 제목 스타일", expanded=False):
    st.markdown("제목은 최대 4줄까지 사용할 수 있습니다.")

    title_font_size = st.slider(
        "제목 폰트 크기", min_value=40, max_value=150, value=90, step=2
    )
    title_outline_width = st.slider(
        "제목 외곽선 두께", min_value=0, max_value=10, value=4
    )
    title_line_spacing = st.slider(
        "제목 줄 간격(픽셀)", min_value=0, max_value=80, value=10
    )
    title_pos_percent = st.slider(
        "제목 블록 상단 위치 (0 = 맨 위, 100 = 맨 아래)",
        min_value=0,
        max_value=40,
        value=10,
    )
    title_top_ratio = title_pos_percent / 100.0

    st.markdown("---")
    st.markdown("**제목 텍스트 & 각 줄 스타일**")

    title_lines = []
    title_aligns = []
    title_text_colors = []
    title_outline_colors = []

    align_label_to_value = {"좌측": "left", "가운데": "center", "우측": "right"}

    for i in range(4):
        st.markdown(f"**제목 {i+1} 줄**")
        line = st.text_input(
            f"제목 {i+1} 줄 내용", key=f"title_line_{i+1}", placeholder="비워두면 사용하지 않습니다."
        )
        align_label = st.selectbox(
            f"정렬 (제목 {i+1} 줄)",
            options=["좌측", "가운데", "우측"],
            index=1,
            key=f"title_align_{i+1}",
        )
        text_color_line = st.color_picker(
            f"글자 색상 (제목 {i+1} 줄)", "#FFFFFF", key=f"title_color_{i+1}"
        )
        outline_color_line = st.color_picker(
            f"외곽선 색상 (제목 {i+1} 줄)", "#000000", key=f"title_outline_color_{i+1}"
        )

        title_lines.append(line)
        title_aligns.append(align_label_to_value[align_label])
        title_text_colors.append(text_color_line)
        title_outline_colors.append(outline_color_line)

        st.markdown("---")

# ---------- 메인 영역 ----------
st.title("🎬 SubKing - 텍스트로 음성 + 자막 영상 만들기")

script = st.text_area(
    "🎧 음성으로 읽어 줄 대본을 입력하세요",
    height=300,
    placeholder="여기에 읽어 줄 문장을 입력해 주세요.",
)

if st.button("🎤 음성 + 자막 영상 생성"):
    if not script.strip():
        st.error("대본을 먼저 입력해 주세요.")
        st.stop()

    with st.status("TTS 생성 중...", expanded=True) as status:
        audio_path = generate_tts(script, voice=selected_voice)
        status.update(label="타임스탬프 분석 중 (Whisper)...", state="running")

        words = extract_word_timestamps(audio_path)
        if not words:
            status.update(
                label="타임스탬프 결과가 비어 있습니다. 텍스트를 다시 확인해 주세요.",
                state="error",
            )
            st.stop()

        status.update(label="영상 렌더링 중 (MoviePy)...", state="running")

        video_path = create_video_with_subtitles(
            audio_path=audio_path,
            words=words,
            video_size=video_size,
            font_size=font_size,
            text_color_hex=text_color,
            outline_color_hex=outline_color,
            outline_width=outline_width,
            y_ratio=y_ratio,
            output_path="subking_result.mp4",
            # 제목 옵션 전달
            title_lines=title_lines,
            title_aligns=title_aligns,
            title_text_colors=title_text_colors,
            title_outline_colors=title_outline_colors,
            title_font_size=title_font_size,
            title_outline_width=title_outline_width,
            title_line_spacing=title_line_spacing,
            title_top_ratio=title_top_ratio,
        )

        if not video_path:
            status.update(label="영상 생성에 실패했습니다.", state="error")
            st.stop()

        status.update(label="완료! 🎉", state="complete")

    st.success("영상이 생성되었습니다.")
    st.video(video_path)

    with open(video_path, "rb") as f:
        st.download_button(
            "📥 영상 다운로드",
            f,
            file_name="subking_result.mp4",
            mime="video/mp4",
        )
