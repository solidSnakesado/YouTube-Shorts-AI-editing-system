# 계층: 비즈니스 로직 계층 (Service 헬퍼) 
# 역할: Phase 2 학습 데이터 생성용 Whisper 전사 + 시간 범위 텍스트 슬라이싱
# 31일차 신규: dataset_builder.py의 300줄 초과 방지를 위해 분리
# 의존: gpu_manager (Whisper 로드/언로드)

"""Phase 2 데이터셋 빌더용 Whisper 전사 헬퍼"""

from pathlib import Path
from loguru import logger

def transcribe_video(video_path: Path) -> list[dict]:
    """
    faster-whisper로 영상 전사, 세그먼트 리스트 반환

    Args:
        video_path: 오디오 트랙이 포함된 영상 파일 경로
    
    Returns:
        [{"start": 0.0, "end": 3.5, "text": "안녕하세요"}, ...]
    """

    from app.core.gpu_manager import load_whisper, unload_model

    logger.info(f"Whisper 전사 시작: {video_path.name}")
    model = load_whisper()
    try:
        segments, info = model.transcribe(
            str(video_path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        result = []
        for seg in segments:
            result.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            })
        logger.info(
            f"Whisper 전사 완료: {len(result)}개 세그먼트 | "
            f"언어: {info.language} ({info.language_probability:.2%})"
        )
        return result
    finally:
        unload_model(model, "Whisper")

def get_text_for_range(segments: list[dict], start: float, end: float) -> str:
    """
    시간 범위에 해당하는 전사 텍스트 추출

    세그먼트의 시작/종료가 범위와 겹치면 포함

    Args:
        segments: transcribe_video() 반환값
        start: 시작 시간 (초)
        end: 종료 시각 (초)

    Returns:
        해당 범위의 전사 텍스트 (공백 구분 결합)
    """

    texts = []
    for seg in segments:
        if seg["end"] > start and seg["start"] < end:
            texts.append(seg["text"])
    return " ".join(texts).strip()