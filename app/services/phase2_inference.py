# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: Phase 2 학습 형식(10초 클립)과 동일한 슬라이딩 윈도우 추론
# 의존: vlm_client (LoRA 로드/생성), frame_extractor, llm_highlight_extractor
# 31일차 신규: 학습-추론 프롬프트 일치를 위해 분리

"""Phase 2 슬라이딩 윈도우 추론 - 10초 클립 단위로 영상을 스캔하여 하이라이트 탐색"""

import asyncio
import json
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.services.frame_extractor import extract_frames
from app.services.llm_highlight_extractor import _extract_json
from app.services.lora_utils import load_lora_model, unload_lora_model, frames_to_pil, lora_generate

# Phase 2 학습과 동일한 설정
WINDOW_SEC = 10
FRAMES_PER_WINDOW = 5
FRAME_RESOLUTION = 336

# 33일차: 회귀 모델 지시문 - relabel_regression.py의 REGRESSION_INSTRUCTION 과 동일해야 함 (학습-추론 일치) 
REGRESSION_INSTRUCTION = (
    "영상 프레임과 전사 텍스트를 분석하여 이 구간이 시청자에게 얼마나 다시 보고 싶은 "
    "구간인지 0.00~1.00 사이 참여도 점수로 예측하세요. "
    "{\"engagement_score\": 점수} 형식의 JSON으로만 반환하세요."
)

# 33일차: 재학습용 프레임 보존 데렉토리 (data/feedback_frames/{video_stem}/w{start}/)
_FEEDBACK_FRAMES_DIR = Path("data/feedback_frames")

def _save_train_frames(images: list, video_stem:str, w_start: float) -> list[str]:
    """33일차: 재학습용 프레임을 디스크에 저장 -> 경로 리스트 반환 (336px JPEG)"""

    frame_dir = _FEEDBACK_FRAMES_DIR / video_stem / f"w{w_start:.0f}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = frame_dir / f"f{i}.jpg"
        img.save(p, quality=90)
        paths.append(str(p))
    return paths

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
        f"{REGRESSION_INSTRUCTION}"
    )

def _parse_engagement(response: str) -> float | None:
    """33일차: 회귀 모델 출력 {"engagement_score": X} 파싱 -> 0.0~1.0 점수 (실패 시 None)"""

    raw = _extract_json(response)
    if not isinstance(raw, dict):       # json.loads가 list를 반환할 수 있음 (구식 빈 리스트 출력 등)
        return None
    score = raw.get("engagement_score")
    if not isinstance(score, (int, float)):
        return None
    return max(0.0, min(1.0, float(score)))

def _expand_clip(w_start: float, w_end: float, target_dur: float | None, duration: float) -> tuple[float, float]:
    """33일차: 탐지 윈도우(10초)를 중심으로 target_duration_sec까지 클립 확장
    탐자 단위는 학습 형식상 10초 고정 -> 출력 클립 길이만 사용자 지정으로 분리"""

    if not target_dur or target_dur <= (w_end - w_start):
        return w_start, w_end
    pad = (target_dur - (w_end - w_start)) / 2
    clip_start = max(0.0, w_start - pad)
    clip_end = min(duration, clip_start + target_dur)
    clip_start = max(0.0, clip_end - target_dur)    # 영상 끝에서 잘렸으면 시작을 당겨 길이 보존
    return round(clip_start, 3), round(clip_end, 3)

def _clip_iou(a: dict, b:dict) -> float:
    """33일차 두 클립 [start_sec, end_sec]의 IOU (겹침 억제용)"""

    inter = min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"])
    if inter <= 0:
        return 0.0
    union = max(a["end_sec"], b["end_sec"]) - min(a["start_sec"], b["start_sec"])
    return inter / union if union > 0 else 0.0

async def run_phase2_inference(source_path: str, transcript_data: dict, max_shorts: int, adapter_path: str) -> list[dict]:
    """Phase 2 슬라이딩 윈도우 추론 - 모델 1회 로드, N개 윈도우 순차 처리
    
    1. 영상을 10초 윈도우로 분할 (스텝: 영상 길이에 따라 10~60초)
    2. 각 윈도우: 5프레임(336px) + Whisper 전사 -> Phase 2 프롬프트
    3. 하이라이트 윈도우 수집 -> hook_score 정렬 -> 상위 max_shorts개 반환
    """

    duration = transcript_data.get("duration_sec", 0)
    title = transcript_data.get("video_title", "알 수 없음")
    target_dur = transcript_data.get("target_duration_sec")     # 33일차: 출력 클립 길이 (None이면 윈도우 그대로)
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

            # 33일차: 회귀 점수 파싱 - 윈도우 자체가 후보 (점수 없으면 건너뜀)
            score = _parse_engagement(result)
            if score is not None:
                # 33일차: 재학습용 프레임 보존 + 메타데이터 태깅
                # 34일차: video_stem을 project_id(부모 디렉토리명)로 - 모든 영상이 source.mp4라
                #   stem이 'source'로 동일 -> 영상 간 프레임 덮어쓰기/오염 + 영상 카운트 1개 버그 수정
                frame_paths = _save_train_frames(images, Path(source_path).parent.name, w_start)
                train_sample = json.dumps({
                    "frame_paths": frame_paths,
                    "prompt": prompt,
                    "window": f"{w_start:.0f}-{w_end:.0f}",
                    "title": title,
                    "transcript": transcript,
                }, ensure_ascii=False)
                # 33일차: 출력 클립은 윈도우 중심으로 target_dur까지 확장 (탐지 단위와 분리)
                clip_start, clip_end = _expand_clip(w_start, w_end, target_dur, duration)
                all_highlights.append({
                    "start_sec": clip_start,
                    "end_sec": clip_end,
                    "hook_score": score,
                    "reason": f"참여도 예측 {score:.2f} (회귀 모델, 탐지 구간 {w_start:.0f}~{w_end:.0F})",
                    "_train_sample_json": train_sample,
                    "_model_version": str(adapter_path),
                })
            
            # 진행 로그 (20% 단위)
            if (i + 1) % max(1, len(windows) // 5) == 0:
                logger.info(f"Phase 2 진행: {i + 1}/{len(windows)}윈도우 | 탐지: {len(all_highlights)}개")
    finally:
        unload_lora_model(model, tokenizer, processor)

    # hook_score 기준 정렬 + greedy 겹침 억제 선택 (33일차: 확장 클립 간 중복 방지)
    all_highlights.sort(key=lambda h: h.get("hook_score", 0), reverse=True)
    result = []
    for cand in all_highlights:
        if len(result) >= max_shorts:
            break
        if any(_clip_iou(cand, sel) > settings.HIGHLIGHT_IOU_THRESHOLD for sel in result):
            continue
        result.append(cand)

    logger.info(f"Phase 2 추론 완료: {len(all_highlights)}개 탐지 -> {len(result)}개 선택")

    return result