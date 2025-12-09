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

# =========================
# Supabase 클라이언트 설정
# =========================
try:
    from supabase import create_client
except ImportError:
    create_client = None

SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# 폰트 (레포 루트에 NanumGothic.ttf 파일이 있다고 가정)
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")

# ====================================
# Session State 기본값
# ====================================
default_state = {
    "ratio_label": "9:16 쇼츠 (1080x1920)",
    "selected_voice": "alloy",
    # 자막
    "sub_font_size": 80,
    "sub_text_color": "#FFFFFF",
    "sub_outline_width": 4,
    "sub_outline_color": "#000000",
    "sub_pos_percent": 80,
    "hide_subtitles": False,
    # 제목
    "title_font_size": 90,
    "title_outline_width": 4,
    "title_line_spacing": 10,
    "title_pos_percent": 10,
    "title_char_spacing": 0,
    "title_raw": "",  # 기본 샘플 제거, 빈 문자열
}
for k, v in default_state.items():
    st.session_state.setdefault(k, v)

# 줄별 스타일(최대 5줄) - 정렬 기본값을 모두 "좌측"
for i in range(5):
    st.session_state.setdefault(f"title_align_label_{i}", "좌측")
    st.session_state.setdefault(f"title_color_{i}", "#FFFFFF")
    st.session_state.setdefault(f"title_outline_color_{i}", "#000000")


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
    자막용 텍스트 이미지 (단어 단위 줄바꿈).
    """
    if not text:
        text = " "

    font = load_font(font_size)
    text_color = hex_to_rgb(text_color_hex)
    outline_color = hex_to_rgb(outline_color_hex)

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

        if outline_width > 0:
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), line, font=font, fill=outline_color)

        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height

    return img


def make_title_line_image(
    text: str,
    font_size: int,
    text_color_hex: str,
    outline_color_hex: str,
    outline_width: int,
    char_spacing: int = 0,
):
    """
    제목 1줄용 이미지 (글자 단위로 가로 간격 조절).
    줄간격/줄 위치는 바깥에서 처리.
    """
    if not text:
        text = " "

    font = load_font(font_size)
    text_color = hex_to_rgb(text_color_hex)
    outline_color = hex_to_rgb(outline_color_hex)

    dummy_img = Image.new("RGBA", (font_size * len(text) * 2, font_size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    char_widths = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        char_widths.append(w)

    total_width = sum(char_widths)
    if len(text) > 1:
        total_width += char_spacing * (len(text) - 1)

    height = font_size + 8

    img = Image.new("RGBA", (max(total_width, 1), height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = 0
    y = 0
    for ch, w in zip(text, char_widths):
        if outline_width > 0:
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), ch, font=font, fill=outline_color)
        draw.text((x, y), ch, font=font, fill=text_color)
        x += w + char_spacing

    return img


# ====================================
# 1) 텍스트 -> 음성 (OpenAI TTS)
# ====================================
def generate_tts(
    text: str,
    voice: str = "alloy",
    output_path: str = "tts_audio.mp3",
) -> str:
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
    min_duration: float = 1.2,
    max_chars: int = 25,
):
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
    y_ratio: float = 0.8,
    line_spacing: int = 8,
):
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
# 3-C) 제목(최대 5줄) 클립 생성
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
    char_spacing: int,
):
    W, H = video_size
    clips = []

    y = int(H * top_ratio)
    margin_x = int(W * 0.1)  # 좌우 여백 10%

    for idx, line in enumerate(title_lines):
        line = line or ""
        if not line.strip():
            continue

        text_color = text_colors[idx] if idx < len(text_colors) else "#FFFFFF"
        outline_color = outline_colors[idx] if idx < len(outline_colors) else "#000000"
        align = aligns[idx] if idx < len(aligns) else "left"

        img = make_title_line_image(
            line,
            font_size=font_size,
            text_color_hex=text_color,
            outline_color_hex=outline_color,
            outline_width=outline_width,
            char_spacing=char_spacing,
        )

        img_array = np.array(img)
        w, h = img.size

        if align == "left":
            x = margin_x
        elif align == "right":
            x = W - margin_x - w
        else:
            x = (W - w) // 2

        clip = (
            ImageClip(img_array)
            .set_duration(duration)
            .set_start(0)
            .set_position((x, y))
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
    hide_subtitles: bool = False,
    # --- 제목 관련 옵션 ---
    title_lines=None,
    title_aligns=None,
    title_text_colors=None,
    title_outline_colors=None,
    title_font_size: int = 80,
    title_outline_width: int = 4,
    title_line_spacing: int = 10,
    title_top_ratio: float = 0.1,
    title_char_spacing: int = 0,
):
    if title_lines is None:
        title_lines = []
    if title_aligns is None:
        title_aligns = []
    if title_text_colors is None:
        title_text_colors = []
    if title_outline_colors is None:
        title_outline_colors = []

    clips = []
    duration = 0.0
    W, H = video_size

    if hide_subtitles:
        norm_words = normalize_words(words)
        if norm_words:
            duration = max(w["end"] for w in norm_words)
        else:
            duration = 0.0

        if duration <= 0:
            return None

        bg = ColorClip(size=(W, H), color=(0, 0, 0), duration=duration)
        clips.append(bg)
    else:
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
    if any((line or "").strip() for line in title_lines):
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
            char_spacing=title_char_spacing,
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
# 5) 미리보기 이미지 생성 (제목 + 자막)
# ====================================
def create_preview_frame(
    video_size=(1080, 1920),
    # 자막 스타일
    sub_font_size: int = 70,
    sub_text_color_hex: str = "#FFFFFF",
    sub_outline_color_hex: str = "#000000",
    sub_outline_width: int = 3,
    sub_y_ratio: float = 0.8,
    sub_sample_text: str = "여기서는 자막이 올라갑니다",
    show_subtitle: bool = True,
    # 제목 스타일
    title_font_size: int = 80,
    title_outline_width: int = 4,
    title_line_spacing: int = 10,
    title_top_ratio: float = 0.1,
    title_char_spacing: int = 0,
    title_lines=None,
    title_text_colors=None,
    title_outline_colors=None,
    title_aligns=None,
):
    if title_lines is None:
        title_lines = []
    if title_text_colors is None:
        title_text_colors = []
    if title_outline_colors is None:
        title_outline_colors = []
    if title_aligns is None:
        title_aligns = []

    W, H = video_size
    bg = Image.new("RGB", (W, H), (0, 0, 0))

    # 1) 제목 부분
    y = int(H * title_top_ratio)
    margin_x = int(W * 0.1)

    for idx, line in enumerate(title_lines):
        line = line or ""
        if not line.strip():
            continue

        text_color = title_text_colors[idx] if idx < len(title_text_colors) else "#FFFFFF"
        outline_color = title_outline_colors[idx] if idx < len(title_outline_colors) else "#000000"
        align = title_aligns[idx] if idx < len(title_aligns) else "left"

        img = make_title_line_image(
            line,
            font_size=title_font_size,
            text_color_hex=text_color,
            outline_color_hex=outline_color,
            outline_width=title_outline_width,
            char_spacing=title_char_spacing,
        )
        w, h = img.size

        if align == "left":
            x = margin_x
        elif align == "right":
            x = W - margin_x - w
        else:
            x = (W - w) // 2

        bg.paste(img, (x, y), img)
        y += title_font_size + title_line_spacing

    # 2) 자막 부분
    if show_subtitle:
        subtitle_img = make_text_image(
            sub_sample_text,
            width=W - 200,
            font_size=sub_font_size,
            text_color_hex=sub_text_color_hex,
            outline_color_hex=sub_outline_color_hex,
            outline_width=sub_outline_width,
            line_spacing=8,
            align="center",
        )

        sw, sh = subtitle_img.size
        y_pos = int(H * sub_y_ratio) - sh // 2
        x_pos = (W - sw) // 2
        bg.paste(subtitle_img, (x_pos, y_pos), subtitle_img)

    # 3) 1/5 크기로 축소
    scale = 0.2
    preview_size = (int(W * scale), int(H * scale))
    preview = bg.resize(preview_size, Image.LANCZOS)
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
    ("9:16 쇼츠 (1080x1920)", "16:9 롱폼 (1920x1080)"),
    key="ratio_label",
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
selected_voice = side.selectbox(
    "🎙 TTS 목소리 선택",
    options=voice_options,
    key="selected_voice",
)

side.markdown("---")

# 자막 스타일
with side.expander("🎨 자막 스타일", expanded=True):
    sub_font_size = st.slider(
        "자막 폰트 크기", min_value=40, max_value=120,
        key="sub_font_size"
    )
    sub_text_color = st.color_picker(
        "자막 글자 색상", key="sub_text_color"
    )

    sub_outline_width = st.slider(
        "텍스트 외곽선 두께", min_value=0, max_value=8,
        key="sub_outline_width"
    )
    sub_outline_color = st.color_picker(
        "외곽선 색상", key="sub_outline_color"
    )

    sub_pos_percent = st.slider(
        "자막 세로 위치 (0 = 맨 위, 100 = 맨 아래)",
        min_value=50,
        max_value=95,
        key="sub_pos_percent",
    )
    sub_y_ratio = st.session_state["sub_pos_percent"] / 100.0

    st.checkbox(
        "자막 숨기기 (미리보기 및 영상에서 숨김)",
        key="hide_subtitles",
    )

# 제목 스타일
with side.expander("📝 제목 스타일", expanded=False):
    st.markdown("제목은 줄바꿈 기준으로 최대 5줄까지 사용할 수 있습니다.")

    # 제목 내용 (한 번에 입력, 줄바꿈으로 구분)
    title_raw = st.text_area(
        "제목 내용 (줄바꿈으로 최대 5줄)",
        key="title_raw",
        height=140,
    )

    title_font_size = st.slider(
        "제목 폰트 크기", min_value=40, max_value=150,
        key="title_font_size"
    )
    title_outline_width = st.slider(
        "제목 외곽선 두께", min_value=0, max_value=10,
        key="title_outline_width"
    )
    title_line_spacing = st.slider(
        "제목 줄 간격(세로, 픽셀)", min_value=0, max_value=80,
        key="title_line_spacing"
    )
    title_char_spacing = st.slider(
        "제목 글자 가로 간격(자간, 픽셀)", min_value=0, max_value=100,
        key="title_char_spacing"
    )
    title_pos_percent = st.slider(
        "제목 블록 상단 위치 (0 = 맨 위, 100 = 맨 아래)",
        min_value=0,
        max_value=40,
        key="title_pos_percent",
    )
    title_top_ratio = st.session_state["title_pos_percent"] / 100.0

    st.markdown("---")
    st.markdown("**각 줄 스타일 (현재 제목 줄 수만큼 표시)**")

    # 현재 제목 줄 수 계산 (최대 5줄)
    raw_lines_for_style = st.session_state["title_raw"].splitlines()
    raw_lines_for_style = raw_lines_for_style[:5]
    num_style_lines = len(raw_lines_for_style)

    align_label_to_value = {"좌측": "left", "가운데": "center", "우측": "right"}

    for i in range(num_style_lines):
        with st.expander(f"제목 {i+1} 줄 스타일", expanded=(i == 0)):
            st.selectbox(
                f"정렬 (제목 {i+1} 줄)",
                options=["좌측", "가운데", "우측"],
                key=f"title_align_label_{i}",
            )
            st.color_picker(
                f"글자 색상 (제목 {i+1} 줄)",
                key=f"title_color_{i}",
            )
            st.color_picker(
                f"외곽선 색상 (제목 {i+1} 줄)",
                key=f"title_outline_color_{i}",
            )

# ---- 프리셋 관리 ----
with side.expander("💾 스타일 프리셋", expanded=False):
    if not supabase:
        st.info(
            "Supabase URL / KEY 를 st.secrets 에 설정하면 "
            "스타일 프리셋 저장/불러오기를 사용할 수 있습니다."
        )
    else:
        preset_name = st.text_input("프리셋 이름", key="preset_name")

        col_save, col_load = st.columns(2)

        with col_save:
            if st.button("현재 스타일 저장", key="save_preset_btn"):
                if not preset_name:
                    st.warning("프리셋 이름을 입력해 주세요.")
                else:
                    ss = st.session_state
                    align_labels = [ss[f"title_align_label_{i}"] for i in range(5)]
                    text_colors = [ss[f"title_color_{i}"] for i in range(5)]
                    outline_colors = [ss[f"title_outline_color_{i}"] for i in range(5)]

                    data = {
                        "ratio_label": ss["ratio_label"],
                        "voice": ss["selected_voice"],
                        "subtitle": {
                            "font_size": ss["sub_font_size"],
                            "text_color": ss["sub_text_color"],
                            "outline_width": ss["sub_outline_width"],
                            "outline_color": ss["sub_outline_color"],
                            "pos_percent": ss["sub_pos_percent"],
                            "hide_subtitles": ss["hide_subtitles"],
                        },
                        "title": {
                            "font_size": ss["title_font_size"],
                            "outline_width": ss["title_outline_width"],
                            "line_spacing": ss["title_line_spacing"],
                            "pos_percent": ss["title_pos_percent"],
                            "char_spacing": ss["title_char_spacing"],
                            "text": ss["title_raw"],
                            "align_labels": align_labels,
                            "text_colors": text_colors,
                            "outline_colors": outline_colors,
                        },
                    }

                    try:
                        supabase.table("subking_presets").upsert(
                            {"name": preset_name, "data": data}
                        ).execute()
                        st.success("프리셋이 저장되었습니다.")
                    except Exception as e:
                        st.error(f"저장 중 오류: {e}")

        with col_load:
            try:
                res = supabase.table("subking_presets").select("name").execute()
                names = sorted({row["name"] for row in res.data}) if res.data else []
            except Exception as e:
                names = []
                st.error(f"프리셋 목록 조회 중 오류: {e}")

            selected_preset_name = st.selectbox(
                "저장된 프리셋",
                options=["선택 안 함"] + names,
                key="selected_preset_name",
            )

            if st.button("프리셋 불러오기", key="load_preset_btn"):
                if selected_preset_name == "선택 안 함":
                    st.warning("불러올 프리셋을 선택해 주세요.")
                else:
                    try:
                        res = (
                            supabase.table("subking_presets")
                            .select("data")
                            .eq("name", selected_preset_name)
                            .single()
                            .execute()
                        )
                        preset = res.data.get("data", {})
                        ss = st.session_state

                        ss["ratio_label"] = preset.get("ratio_label", ss["ratio_label"])
                        ss["selected_voice"] = preset.get("voice", ss["selected_voice"])

                        sub = preset.get("subtitle", {})
                        ss["sub_font_size"] = sub.get("font_size", ss["sub_font_size"])
                        ss["sub_text_color"] = sub.get("text_color", ss["sub_text_color"])
                        ss["sub_outline_width"] = sub.get("outline_width", ss["sub_outline_width"])
                        ss["sub_outline_color"] = sub.get("outline_color", ss["sub_outline_color"])
                        ss["sub_pos_percent"] = sub.get("pos_percent", ss["sub_pos_percent"])
                        ss["hide_subtitles"] = sub.get("hide_subtitles", ss["hide_subtitles"])

                        title = preset.get("title", {})
                        ss["title_font_size"] = title.get("font_size", ss["title_font_size"])
                        ss["title_outline_width"] = title.get("outline_width", ss["title_outline_width"])
                        ss["title_line_spacing"] = title.get("line_spacing", ss["title_line_spacing"])
                        ss["title_pos_percent"] = title.get("pos_percent", ss["title_pos_percent"])
                        ss["title_char_spacing"] = title.get("char_spacing", ss["title_char_spacing"])
                        ss["title_raw"] = title.get("text", ss["title_raw"])

                        align_labels = title.get("align_labels", [])
                        text_colors = title.get("text_colors", [])
                        outline_colors = title.get("outline_colors", [])

                        for i in range(5):
                            if i < len(align_labels):
                                ss[f"title_align_label_{i}"] = align_labels[i]
                            if i < len(text_colors):
                                ss[f"title_color_{i}"] = text_colors[i]
                            if i < len(outline_colors):
                                ss[f"title_outline_color_{i}"] = outline_colors[i]

                        st.success("프리셋을 적용했습니다.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"프리셋 불러오기 중 오류: {e}")

# ---------- 메인 영역 ----------
st.title("🎬 SubKing - 텍스트로 음성 + 자막 영상 만들기")

script = st.text_area(
    "🎧 음성으로 읽어 줄 대본을 입력하세요",
    height=100,  # 1/3 정도로 줄임
    placeholder="여기에 읽어 줄 문장을 입력해 주세요.",
)

# ---- 미리보기 (제목 + 자막) ----
st.markdown("### 🔍 미리보기 (제목 + 자막 스타일)")

# 제목 라인 파싱 (최대 5줄, 샘플 없이 실제 입력만 사용)
raw_lines = st.session_state["title_raw"].splitlines()
title_lines = raw_lines[:5]  # 그대로 사용
num_preview_lines = len(title_lines)

title_aligns = []
title_text_colors = []
title_outline_colors = []

align_label_to_value = {"좌측": "left", "가운데": "center", "우측": "right"}

for i in range(num_preview_lines):
    align_label = st.session_state[f"title_align_label_{i}"]
    title_aligns.append(align_label_to_value.get(align_label, "left"))
    title_text_colors.append(st.session_state[f"title_color_{i}"])
    title_outline_colors.append(st.session_state[f"title_outline_color_{i}"])

preview_img = create_preview_frame(
    video_size=video_size,
    # 자막
    sub_font_size=sub_font_size,
    sub_text_color_hex=sub_text_color,
    sub_outline_color_hex=sub_outline_color,
    sub_outline_width=sub_outline_width,
    sub_y_ratio=sub_y_ratio,
    sub_sample_text="여기서는 자막이 올라갑니다",
    show_subtitle=not st.session_state["hide_subtitles"],
    # 제목
    title_font_size=title_font_size,
    title_outline_width=title_outline_width,
    title_line_spacing=title_line_spacing,
    title_top_ratio=title_top_ratio,
    title_char_spacing=title_char_spacing,
    title_lines=title_lines,
    title_text_colors=title_text_colors,
    title_outline_colors=title_outline_colors,
    title_aligns=title_aligns,
)

st.image(preview_img, caption="현재 제목 + 자막 스타일 미리보기", use_container_width=False)

st.markdown("---")

# ---- 영상 생성 버튼 ----
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
            font_size=sub_font_size,
            text_color_hex=sub_text_color,
            outline_color_hex=sub_outline_color,
            outline_width=sub_outline_width,
            y_ratio=sub_y_ratio,
            output_path="subking_result.mp4",
            hide_subtitles=st.session_state["hide_subtitles"],
            # 제목 옵션
            title_lines=title_lines,
            title_aligns=title_aligns,
            title_text_colors=title_text_colors,
            title_outline_colors=title_outline_colors,
            title_font_size=title_font_size,
            title_outline_width=title_outline_width,
            title_line_spacing=title_line_spacing,
            title_top_ratio=title_top_ratio,
            title_char_spacing=title_char_spacing,
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
