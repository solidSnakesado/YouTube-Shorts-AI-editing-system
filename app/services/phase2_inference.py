# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: Phase 2 학습 형식(10초 클립)과 동일한 슬라이딩 윈도우 추론
# 의존: vlm_client (LoRA 로드/생성), frame_extractor, llm_highlight_extractor
# 31일차 신규: 학습-추론 프롬프트 일치를 위해 분리

"""Phase 2 슬라이딩 윈도우 추론 - 10초 클립 단위로 영상을 스캔하여 하이라이트 탐색"""

import asyncio
from pathlib import Path

from loguru import logger

from app.services.frame_extractor import extract_frames
from app.services.llm_highlight_extractor import parse_highlights
from app.services.lora_utils import load_lora_model, unload_lora_model, frames_to_pil, lora_generate

# Phase 2 학습과 동일한 설정
WINDOW_SEC = 10
FRAMES_PER_WINDOW = 5
FRAME_RESOLUTION = 336

def _compute_step_sec(duration: float) -> int:
    """영상 길이에 따라 슬라이딩 윈도우 스텝 크기 결정
    짧은 영상: 전체 커버, 긴 영상: 넓은 간격으로 효율적 탐색"""
    
    if duration < 300:
        return 10
    elif duration < 3600:
        return 30
    else:
        return 60

def _get_transcript_for_range(transcript_data: dict, start: float, end: float) -> str:
    """전사 데이터에서 특정 시간 범위의 텍스트 추출"""
    
    segments = transcript_data.get("segments", [])
    texts = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        if seg_end > start and seg_start < end:
            text = seg.get("text", "").strip()
            if text: texts.append(text)

    return " ".join(texts)

def _build_phase2_prompt(title: str, duration: float, clip_start: float, clip_end: float, transcript: str) -> str:
    """Phase 2 학습 프롬프트와 동일한 형식으로 생성"""
    
    transcript_line = f"음성 전사: {transcript}\n" if transcript else ""
    return (
        f"영상: {title}\n"
        f"전체 길이: {duration:.0f}초\n"
        f"클립 구간: {clip_start:.1f}초 ~ {clip_end:.1f}초\n"
        f"{transcript_line}\n"
        f"영상 프레임과 전사 텍스트를 분석하여 쇼츠 하이라이트 구간을 "
        f"JSON으로 추출하세요. 하이라이트가 없으면 빈 리스트로 반환하세요."
    )

async def run_phase2_inference(source_path: str, transcript_data: dict, max_shorts: int, adapter_path: str) -> list[dict]:
    """Phase 2 슬라이딩 윈도우 추론 - 모델 1회 로드, N개 윈도우 순차 처리
    
    1. 영상을 10초 윈도우로 분할 (스텝: 영상 길이에 따라 10~60초)
    2. 각 윈도우: 5프레임(336px) + Whisper 전사 -> Phase 2 프롬프트
    3. 하이라이트 윈도우 수집 -> hook_score 정렬 -> 상위 max_shorts개 반환
    """

    duration = transcript_data.get("duration_sec", 0)
    title = transcript_data.get("video_title", "알 수 없음")
    step_sec = _compute_step_sec(duration)

    # 윈도우 생성
    windows = []
    t = 0.0
    while t + WINDOW_SEC <= duration:
        windows.append((t, t + WINDOW_SEC))
        t += step_sec
    if not windows:
        windows.append((0, min(WINDOW_SEC, duration)))

    logger.info(f"Phase 2 추론 시작 | {len(windows)}개 윈도우 | 스탭: {step_sec}초 | 영상: {duration:.0f}초")

    # 모델 1회 로드
    model, tokenizer, processor = load_lora_model(adapter_path)
    loop = asyncio.get_event_loop()
    all_highlights = []

    try:
        for i, (w_start, w_end) in enumerate(windows):
            # 프레임 추출 (5장, 336px, 2초 간격)
            frames = await extract_frames(
                Path(source_path),
                interval_sec=2.0,
                start_sec=w_start,
                end_sec=w_end,
                max_frames=FRAMES_PER_WINDOW,
                resolution=FRAME_RESOLUTION,
            )
            if not frames:
                continue

            # 전사 텍스트 추출
            transcript = _get_transcript_for_range(transcript_data, w_start, w_end)

            # Phase 2 학습과 동일한 프롬프트
            prompt = _build_phase2_prompt(title, duration, w_start, w_end, transcript)

            # LoRA 추론
            images = frames_to_pil(frames, max_count=FRAMES_PER_WINDOW)
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})

            result = await loop.run_in_executor(
                None, lora_generate, model, tokenizer, processor,
                [{"role": "user", "content": content}], 512, 0.3,
            )

            # 결과 파싱
            highlights = parse_highlights(result, duration, max_shorts)
            if highlights:
                for h in highlights:
                    h["_window"] = f"{w_start:.0f}-{w_end:.0f}"
                all_highlights.extend(highlights)
            
            # 진행 로그 (20% 단위)
            if (i + 1) % max(1, len(windows) // 5) == 0:
                logger.info(f"Phase 2 진행: {i + 1}/{len(windows)}윈도우 | 탐지: {len(all_highlights)}개")
    finally:
        unload_lora_model(model, tokenizer, processor)

    # hook_score 기준 정렬 + 상위 선택
    all_highlights.sort(key=lambda h: h.get("hook_score", 0), reverse=True)
    result = all_highlights[:max_shorts]

    logger.info(f"Phase 2 추론 완료: {len(all_highlights)}개 탐지 -> {len(result)}개 선택")

    return result