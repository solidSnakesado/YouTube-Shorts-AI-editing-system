# 계층: 비즈니스 로직 계층 (Service 헬퍼) 
# 역할: 히트맵 비피크 구간에서 네거티브 학습 샘플 생성 (Phase 2 생성기용)
# 의존: dataset_builder (다운로드/프레임 추출), dataset_transcriber (Whisper), dataset_utils
# 31일차 신규: 네거티브 샘플 추가를 위한 전용 빌더

"""네거티브 데이터셋 빌더 - 비피크 구간 10초 클립에서 빈 하이라이트 샘플 생성"""

import asyncio
import json
import random
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.services.frame_extractor import extract_frames
from app.services.dataset_utils import load_processed_ids, refresh_firefox_cookies
from app.services.dataset_transcriber import transcribe_video, get_text_for_range, init_whisper, cleanup_whisper

# Phase 2 학습과 동일한 설정
CLIP_SEC = 10
FRAMES_PER_CLIP = 5
FRAME_RESOLUTION = 336
INSTRUCTION = (
    "영상 프레임과 전사 텍스트를 분석하여 쇼츠 하이라이트 구간을 JSON으로 추출하세요. "
    "하이라이트가 없으면 빈 리스트를 반환하세요."
)
NEGATIVE_OUTPUT = '{"highlights": []}'

def _pick_negative_clips(peaks: list[dict], duration: float, count: int) -> list[tuple[float, float]]:
    """피크와 겹치지 않는 10초 클립을 랜덤 선택"""

    peak_secs = set()
    for p in peaks:
        for s in range(max(0, int(p["start_sec"]) - 5), int(p["end_sec"]) + 6):
            peak_secs.add(s)

    candidates = []
    t = 0.0
    while t + CLIP_SEC <= duration:
        clip_secs = set(range(int(t), int(t + CLIP_SEC) + 1))
        if not clip_secs & peak_secs:
            candidates.append(t)
        t += CLIP_SEC

    if not candidates:
        return []
    chosen = random.sample(candidates, min(count, len(candidates)))
    return [(s, s + CLIP_SEC) for s in chosen]

async def build_negative_dataset(heatmap_path: Path, pos_jsonl_path: Path, output_path: Path, neg_per_video: int = 0, seed: int = 42) -> dict:
    """네거티브 데이터셋 빌드 메인 함수
    
    Args:
        heatmap_path: 히트맵 JSONL
        pos_jsonl_path: 네거티브 JSONL 출력 경로
        neg_per_video: 영상당 네거티브 클립 수 (0이면 포지티브 수에 맞춤)
        seed: 랜덤 시드
    """

    random.seed(seed)
    stats = {"total": 0, "processed": 0, "skipped": 0, "samples": 0}

    # 포지티브 JSONL에서 처리된 영상 + 영상당 포지티브 수 집계
    pos_counts = _count_pos_per_video(pos_jsonl_path)
    if not pos_counts:
        logger.error("포지티브 JSONL에 처리된 영상이 없습니다.")
        return stats
    
    # 이미 생성된 네거티브 스킵 (이어서 진행)
    neg_done = load_processed_ids(output_path)
    if neg_done:
        logger.info(f"이어서 진행 | 기처리 {len(neg_done)}개 스킵")

    # 히트맵 로드
    videos = _load_heatmap(heatmap_path)
    stats["total"] = len(videos)

    # Whisper 초기화
    init_whisper()

    try:
        for idx, video in enumerate(videos, 1):
            vid = video["video_id"]
            if vid not in pos_counts or vid in neg_done:
                continue

            duration = video["duration_sec"]
            peaks = video.get("peak_segments", [])
            title = video.get("title", "")

            # 영상당 네거티브 수 결정
            count = neg_per_video if neg_per_video > 0 else pos_counts[vid]

            # 비피크 클립 선택
            clips = _pick_negative_clips(peaks, duration, count)
            if not clips:
                stats["skipped"] += 1
                continue

            logger.info(f"[{idx}] 네거티브 처리: {vid} | {len(clips)}개 클립")

            # 다운로드
            temp_dir = Path("temp/finetune")
            temp_dir.mkdir(parents=True, exist_ok=True)
            video_path = temp_dir / f"{vid}.mp4"

            try:
                await _download_video(vid, video_path)
                if not video_path.is_file():
                    stats["skipped"] += 1
                    continue

                # Whisper 전사 (전체 영상 1회)
                transcript = transcribe_video(video_path) if settings.P2_WHISPER_TEXT else []

                # 각 클립에 대해 네거티브 샘플 생성
                samples = []
                for clip_start, clip_end in clips:
                    frames = await extract_frames(
                        video_path, interval_sec=2.0,
                        start_sec=clip_start, end_sec=clip_end,
                        max_frames=FRAMES_PER_CLIP, resolution=FRAME_RESOLUTION,
                        save_dir=Path(settings.FINETUNE_OUTPUT_DIR) / "frames" / f"{vid}_{int(clip_start)}_neg"
                    )
                    if not frames:
                        continue

                    frame_paths = [str(Path(r["path"]).relative_to(Path.cwd())) for r in frames]
                    text = get_text_for_range(transcript, clip_start, clip_end) if transcript else ""

                    meta = {
                        "video_id": vid, "video_title": title,
                        "duration_sec": duration,
                        "clip_start": clip_start, "clip_end": clip_end,
                        "highlight_count": 0,
                    }
                    if text:
                        meta["transcript"] = text

                    samples.append({
                        "instruction": INSTRUCTION,
                        "images": frame_paths,
                        "metadata": meta,
                        "output": NEGATIVE_OUTPUT,
                    })

                # 저장
                _append_samples(output_path, samples)
                stats["processed"] += 1
                stats["samples"] += len(samples)
                logger.info(f"[{idx}] 완료: {vid} | {len(samples)}개 네거티브")

            except Exception as e:
                logger.error(f"[{idx}] 실패: {vid} | {e}")
                stats["skipped"] += 1
            finally:
                if video_path.is_file():
                    video_path.unlink()
    finally:
        cleanup_whisper()

    logger.info(f"네거티브 빌드 완료 | 처리: {stats['processed']} | 샘플: {stats['samples']}")
    return stats

# --------------------------------------------------------------
# 헬퍼 함수
# --------------------------------------------------------------

def _count_pos_per_video(jsonl_path: Path) -> dict[str, int]:
    """포지티브 JSONL에서 영상당 샘플 수 집계"""

    counts: dict[str, int] = {}
    if not jsonl_path.is_file():
        return counts
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                vid = rec.get("metadata", {}).get("video_id")
                if vid and rec.get("instruction") != "skip":
                    counts[vid] = counts.get(vid, 0) + 1
            except Exception:
                pass
    return counts

def _load_heatmap(path: Path) -> list[dict]:
    """히트맵 JSONL 로드"""

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

async def _download_video(video_id: str, output_path: Path) -> None:
    """yt-dlp 전체 다운로드 (오디오 포함 - Whisper 전사용)"""       # 31일차 수정: 오디오 포함 포맷

    from app.services.dataset_utils import refresh_firefox_cookies
    cookie_file = Path("data/youtube_cookies.txt")
    refresh_firefox_cookies(str(cookie_file))
    cookie_opts = ["--cookies", str(cookie_file)] if cookie_file.is_file() else []

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp", "-f", "bestvideo[height<=144]+bestaudio/worst[ext=mp4]/worst",                    # 31일차 수정: 160/394 -> worst (오디오 포함)
        "--merge-output-format", "mp4",                                                             # 추가: FFmpeg 병합 시 mp4로 출력
        "-o", str(output_path), "--no-playlist",
        "--socket-timeout", str(settings.HEATMAP_REQUEST_TIMEOUT_SEC),
        "--no-warnings", "--js-runtimes", "node",
        *cookie_opts, url,
    ]
    logger.info(f"영상 다운로드 시작: {video_id}")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp 실패 (rc={proc.returncode}): {stderr.decode(errors='replace')[:300]}")
    logger.info(f"영상 다운로드 완료: {video_id}")

def _append_samples(output_path: Path, samples: list[dict]) -> None:
    """JSONL에 샘플 추가"""

    if not samples:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")