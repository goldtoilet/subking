import os
import streamlit as st
from openai import OpenAI
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ColorClip

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

# 폰트 (레포 루트에 NanumGothic.ttf 가 있다고 가정)
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")


# ====================================
# 1) 텍스트 -> 음성 (OpenAI TTS)
# ====================================
def generate_tts(text: str, output_path: str = "tts_audio.mp3") -> str:
    """
    텍스트를 OpenAI TTS로 mp3 파일로 저장.
    format 파라미터는 사용하지 않고, 응답 bytes 그대로 저장.
    """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )

    # 응답 객체에서 raw bytes 읽기
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

    # SDK 버전에 따라 pydantic 객체 / dict 둘 다 대응
    words = getattr(transcript, "words", None)
    if words is None and isinstance(transcript, dict):
        words = transcript.get("words", [])

    if words is None:
        words = []

    return words


# ====================================
# 3) 타임스탬프 기반 자막 + 배경 클립 생성
# ====================================
def build_video_clips_from_words(words, video_size=(1080, 1920)):
    """
    Whisper 단어 리스트로부터 자막 텍스트 클립 + 배경 클립 생성.
    """
    W, H = video_size
    clips = []

    if not words:
        return clips, 0.0

    # 전체 길이
    last_end = max((w.end if hasattr(w, "end") else w["end"]) for w in words)

    # 배경(검정 화면)
    bg = ColorClip(size=(W, H), color=(0, 0, 0), duration=last_end)
    clips.append(bg)

    # 각 단어별 자막 클립
    for w in words:
        if hasattr(w, "word"):
            txt = w.word
            start = w.start
            end = w.end
        else:
            txt = w["word"]
            start = w["start"]
            end = w["end"]

        if end <= start:
            continue

        # 폰트 경로가 있으면 사용, 없으면 시스템 기본 폰트
        font_arg = FONT_PATH if os.path.isfile(FONT_PATH) else "Arial-Bold"

        text_clip = (
            TextClip(
                txt,
                font=font_arg,
                fontsize=70,
                color="white",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(W - 200, None),
            )
            .set_position(("center", H - 300))
            .set_start(start)
            .set_duration(end - start)
        )

        clips.append(text_clip)

    return clips, last_end


# ====================================
# 4) 음성 + 자막 -> mp4 영상 만들기
# ====================================
def create_video_with_subtitles(
    audio_path: str, words, output_path: str = "subking_result.mp4"
):
    clips, duration = build_video_clips_from_words(words)
    if duration <= 0:
        return None

    video = CompositeVideoClip(clips)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)

    # mp4로 렌더링
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
# 5) Streamlit UI
# ====================================
st.set_page_config(page_title="SubKing", page_icon="🎬", layout="centered")

st.title("🎬 SubKing - 텍스트로 음성 + 자막 영상 만들기")

script = st.text_area(
    "대본을 입력하세요",
    height=250,
    placeholder="여기에 읽어 줄 문장을 입력해 주세요.",
)

if st.button("🎤 음성 + 자막 영상 생성"):
    if not script.strip():
        st.error("대본을 먼저 입력해 주세요.")
        st.stop()

    with st.status("TTS 생성 중...", expanded=True) as status:
        # 1) 음성 생성
        audio_path = generate_tts(script)
        status.update(label="타임스탬프 분석 중 (Whisper)...", state="running")

        # 2) 타임스탬프 추출
        words = extract_word_timestamps(audio_path)
        if not words:
            status.update(
                label="타임스탬프 결과가 비어 있습니다. 텍스트를 다시 확인해 주세요.",
                state="error",
            )
            st.stop()

        status.update(label="영상 렌더링 중 (MoviePy)...", state="running")

        # 3) 영상 생성
        video_path = create_video_with_subtitles(audio_path, words)

        if not video_path:
            status.update(label="영상 생성에 실패했습니다.", state="error")
            st.stop()

        status.update(label="완료! 🎉", state="complete")

    st.success("영상이 생성되었습니다.")
    st.video(video_path)

    # 다운로드 버튼
    with open(video_path, "rb") as f:
        st.download_button(
            "📥 영상 다운로드",
            f,
            file_name="subking_result.mp4",
            mime="video/mp4",
        )
