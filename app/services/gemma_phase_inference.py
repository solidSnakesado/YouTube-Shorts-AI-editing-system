# 계층: 비즈니스 로직 계층 (Service)
# 50일차 신규 (전체 신규) — 레포 경로: app/services/gemma_phase_inference.py
#   수정 5회(56일차): 추론 종료 시 VRAM 미회수 버그 — finally 블록에 gc.collect 선행
#     (release_vram) + collate/norm 참조 제거 + 잔량 로그 (수정본 기준 L177~188)
#   수정 4회(52일차): 계층 발행 태깅 - confidence_tier + reason 프리픽스 (수정본 기준 L215~234)
#   수정 1회(50일차): gemma_sample import를 app.services.gemma_sample로 교정 (L27~29)
#   수정 2회(50일차): 보존 경로 키 parent.name -> video.stem (다영상 덮어쓰기 방지)
#   수정 3회(50일차): 파이프라인에서 stem="source" 충돌 발견 -> parent.name+stem 결합 (L116~120)
# 역할: Gemma round12 e2e 회귀 어댑터의 파이프라인 추론
#   영상 -> 30s 슬라이딩 윈도우 -> [1fps 프레임 + 오디오] -> hook_score(회귀 헤드)
#   타임스탬프는 클립 윈도우에서 재구성한다 (모델 출력 아님 — 로드맵 3단계 확정 사항)
# 계약: phase2_inference.run_phase2_inference와 동일 출력 dict 키
#   (title/start_sec/end_sec/hook_score/reason/_train_sample_json/_model_version/is_exploration)
#   -> analysis_service/feedback_service/measure_ok_rate 기존 흐름 재사용 가능
# 의존: scripts/gemma_e2e_model.py(수정 7회), scripts/gemma_e2e_collate.py,
#   레포 루트 gemma_collate.py(50일차 수정 1회 — 30s 무음 패딩), gemma_sample.py

"""Gemma 4 E4B 파이프라인 추론 - 영상에서 hook_score 상위 하이라이트 추출"""

import asyncio
import json
import random
import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.core.gemma_config import gemma_settings
from app.services.frame_extractor import extract_frames
from app.services.gemma_audio_extractor import extract_audio_segment
from app.services.phase2_inference import _clip_iou, _expand_clip
from app.services.gemma_sample import (
    HIGHLIGHT_INSTRUCTION, build_gemma_sample, build_highlight_output,
)

# 50일차: scripts/ 모듈(gemma_e2e_model, gemma_e2e_collate)은 패키지가 아니므로 경로 추가
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# 재학습용 미디어 영구 보존 경로 (phase2의 feedback_frames와 동일 취지, Gemma 격리)
_FEEDBACK_MEDIA_DIR = Path("./data/feedback_media_gemma")


def _step_sec(duration: float) -> int:
    """50일차: 영상 길이별 윈도우 스텝 (윈도우는 30s 고정 — 학습 클립과 동일)"""

    if duration <= 600:
        return 30           # 10분 이하: 전 구간 무결 커버
    if duration <= 3600:
        return 60           # 1시간 이하: 절반 커버 (속도 균형)
    return 120              # 초장편: 1/4 커버


def _load_stack(adapter_dir: str):
    """50일차: 추론 스택 1회 로드 -> (model, collate_fn, norm) — blocking(스레드에서 호출)"""

    from gemma_e2e_collate import build_e2e_collate_fn          # scripts/
    from gemma_e2e_model import load_model_for_infer, load_norm_stats  # scripts/

    model, processor = load_model_for_infer(adapter_dir)        # PLE CPU 상주 포함
    model.head.to("cuda")
    collate = build_e2e_collate_fn(processor, max_frames=8)     # 학습과 동일 8프레임 샘플링
    norm = load_norm_stats(adapter_dir)
    return model, collate, norm


def _score_one(model, collate, norm, sample: dict) -> float:
    """50일차: 1윈도우 hook_score — blocking(스레드에서 호출)"""

    import torch

    batch = collate([sample])
    batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    with torch.no_grad():
        p = model(batch).float().cpu().numpy()
    return float(p[0] * norm["sd"] + norm["mu"])


async def _extract_window(video: Path, w_start: float, w_end: float, out_dir: Path):
    """50일차: 1윈도우 미디어 추출 -> (frame_paths, audio_path|None). 영구 경로에 저장"""

    frames = await extract_frames(
        video_path=video,
        interval_sec=1.0 / max(1, gemma_settings.GEMMA_FRAME_FPS),
        max_frames=gemma_settings.GEMMA_AUDIO_MAX_SEC,          # 30장 상한 (1fps x 30s)
        resolution=gemma_settings.GEMMA_FRAME_RESOLUTION,
        start_sec=w_start,
        end_sec=w_end,
        save_dir=out_dir / "frames",
    )
    frame_paths = [f["path"] for f in frames if f.get("path")]
    audio_path = await extract_audio_segment(
        video, w_start, w_end, out_dir / "audio.wav",
    )
    return frame_paths, (str(audio_path) if audio_path else None)


async def run_gemma_inference(
    source_path: str,
    transcript_data: dict,
    max_shorts: int,
    adapter_dir: str,
) -> list[dict]:
    """Gemma 슬라이딩 윈도우 추론 - 모델 1회 로드, N개 윈도우 순차 처리

    1. 영상을 30초 윈도우로 분할 (스텝: 길이에 따라 30/60/120초)
    2. 각 윈도우: 1fps 프레임 + 30s 오디오 -> e2e 회귀 -> hook_score
    3. hook_score 정렬 -> IoU 억제 + 탐색 쿼터로 상위 max_shorts개 반환
    """

    video = Path(source_path)
    duration = float(transcript_data.get("duration_sec", 0) or 0)
    title = transcript_data.get("video_title", "알 수 없음")
    target_dur = transcript_data.get("target_duration_sec")
    # 50일차 수정 2회: 보존 경로 키를 영상 파일명(stem=YouTube ID)으로 — parent.name은
    #   공용 폴더(data/videos/)에서 모든 영상이 같은 경로에 덮어써 충돌
    # 50일차 수정 3회: 파이프라인 경로(temp/<프로젝트ID>/source.mp4)에서는 stem이 전부
    #   "source"라 수정 2회가 오히려 충돌 유발 -> 폴더명+파일명 결합으로 양쪽 모두 유일화
    video_stem = f"{video.parent.name}_{video.stem}"

    step = _step_sec(duration)
    windows: list[tuple[float, float]] = []
    s = 0.0
    while s < duration:
        e = min(s + gemma_settings.GEMMA_AUDIO_MAX_SEC, duration)
        if e - s >= gemma_settings.GEMMA_AUDIO_MAX_SEC / 2:     # 잔여 15s 미만 스킵 (빌더와 동일)
            windows.append((s, e))
        s += step
    logger.info(f"Gemma 추론 시작: {len(windows)}윈도우 (스텝 {step}s) | {video}")

    loop = asyncio.get_running_loop()
    model, collate, norm = await loop.run_in_executor(None, _load_stack, adapter_dir)

    all_highlights: list[dict] = []
    try:
        for i, (w_start, w_end) in enumerate(windows):
            out_dir = _FEEDBACK_MEDIA_DIR / video_stem / f"w{w_start:.0f}"
            try:
                frame_paths, audio_path = await _extract_window(video, w_start, w_end, out_dir)
            except Exception as exc:                            # 추출 실패 윈도우는 스킵
                logger.warning(f"윈도우 {w_start:.0f}s 미디어 추출 실패: {exc}")
                continue
            if not frame_paths or audio_path is None:           # 무음/무프레임 -> 학습 분포 밖
                logger.debug(f"윈도우 {w_start:.0f}s 스킵 (프레임 {len(frame_paths)}개, 오디오 {audio_path})")
                continue

            sample = build_gemma_sample(
                frame_paths=frame_paths,
                audio_path=audio_path,
                instruction=HIGHLIGHT_INSTRUCTION,
                output_json=build_highlight_output(0.0),        # 더미 타겟 (collate 스키마용, 미사용)
                metadata={"video_id": video_stem, "start_sec": w_start, "end_sec": w_end},
            )
            try:
                score = await loop.run_in_executor(
                    None, _score_one, model, collate, norm, sample)
            except Exception as exc:
                logger.warning(f"윈도우 {w_start:.0f}s 추론 실패: {exc}")
                continue

            clip_start, clip_end = _expand_clip(w_start, w_end, target_dur, duration)
            all_highlights.append({
                "title": title,
                "start_sec": clip_start,
                "end_sec": clip_end,
                "hook_score": round(score, 4),
                "reason": f"참여도 예측 {score:.2f} (Gemma e2e, 탐지 구간 {w_start:.0f}~{w_end:.0f})",
                "_train_sample_json": json.dumps(sample, ensure_ascii=False),
                "_model_version": str(adapter_dir),
                "is_exploration": False,
            })
            if (i + 1) % max(1, len(windows) // 5) == 0:
                logger.info(f"Gemma 진행: {i + 1}/{len(windows)}윈도우 | 점수 수집: {len(all_highlights)}개")
    finally:
        # 56일차 수정 5회: 상주 VRAM 미회수 버그 수정 — 기존 del model+empty_cache만으로는
        #   순환 참조(PEFT/unsloth)가 남아 캐시 반환이 무효 -> 다음 영상 전사(Whisper)가
        #   공유 메모리 spillover로 수십 분 지연. gc.collect 선행 + collate/norm 참조 제거
        #   + 잔량 로그로 회수 여부를 실측 가시화 (gpu_manager 표준 절차 재사용)
        from app.core.gpu_manager import get_vram_status, release_vram
        del model, collate, norm
        release_vram()                      # gc.collect() + torch.cuda.empty_cache()
        vram = get_vram_status()
        logger.info(
            f"Gemma 스택 언로드 | VRAM allocated {vram['allocated_mb']}MB / "
            f"reserved {vram['reserved_mb']}MB")

    # quota 선택: 활용(top-K, IoU 억제) + 탐색(저득점 랜덤) — phase2 36일차(F)와 동일 로직
    all_highlights.sort(key=lambda h: h.get("hook_score", 0), reverse=True)
    explore_n = max(0, settings.EXPLORATION_COUNT)
    exploit_n = max(0, max_shorts - explore_n)

    def _fits(cand: dict, chosen: list) -> bool:
        return all(_clip_iou(cand, sel) <= settings.HIGHLIGHT_IOU_THRESHOLD for sel in chosen)

    result: list[dict] = []
    for cand in all_highlights:                                 # 1) 활용
        if len(result) >= exploit_n:
            break
        if _fits(cand, result):
            result.append(cand)
    chosen_ids = {id(c) for c in result}
    if explore_n > 0:                                           # 2) 탐색
        pool = [c for c in all_highlights
                if id(c) not in chosen_ids
                and c.get("hook_score", 0) >= settings.EXPLORATION_MIN_SCORE]
        random.shuffle(pool)
        for cand in pool:
            if len(result) >= max_shorts:
                break
            if _fits(cand, result):
                cand["is_exploration"] = True
                result.append(cand)
                chosen_ids.add(id(cand))
    for cand in all_highlights:                                 # 3) 부족분 활용으로 보충
        if len(result) >= max_shorts:
            break
        if id(cand) not in chosen_ids and _fits(cand, result):
            result.append(cand)
            chosen_ids.add(id(cand))

    # 52일차: 계층 발행 태깅 - 활용 픽을 고신뢰(≥임계, 발행 우선)/보충으로 구분
    #   reason 프리픽스로 DB 스키마 무수정 가시화 + confidence_tier 키 (발행 필터용)
    thr = gemma_settings.GEMMA_PUBLISH_THRESHOLD
    tier_ko = {"high": "고신뢰", "fill": "보충", "explore": "탐색"}
    for cand in result:
        if cand.get("is_exploration"):
            tier = "explore"
        elif cand.get("hook_score", 0) >= thr:
            tier = "high"
        else:
            tier = "fill"
        cand["confidence_tier"] = tier
        cand["reason"] = f"[{tier_ko[tier]}] {cand['reason']}"

    explore_picked = sum(1 for c in result if c.get("is_exploration"))
    high_picked = sum(1 for c in result if c.get("confidence_tier") == "high")
    logger.info(
        f"Gemma 추론 완료: {len(all_highlights)}개 점수 -> {len(result)}개 선택 "
        f"(활용 {len(result) - explore_picked} + 탐색 {explore_picked} | "
        f"고신뢰 {high_picked}, 임계 {thr})")
    return result