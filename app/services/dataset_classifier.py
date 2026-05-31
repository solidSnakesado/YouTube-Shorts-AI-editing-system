# 계층: 비즈니스 로직 계층 (Service 헬퍼) 
# 역할: dataset_builder.py classifier 모드 샘플 생성 로직 분리
# 27일차 신규: 피크별 개별 클립 다운로드 + 상대 타임스탬프 변환
# 의존: dataset_builder.py, frame_extractor.py

"""판별기 모드 샘플 생성 - 피크/네거티브 구간별 클립 다운로드 및 프레임 추출"""

from pathlib import Path

from loguru import logger

async def build_classifier_samples(
    vid: str,
    title: str,
    duration: float,
    peaks: list[dict],
    temp_dir: Path,
    negative_ratio: float,
    stats: dict,
    download_fn,
    extract_fn,
    metadata_fn,
    pick_neg_fn,
) -> list[dict]:
    """
    classifier 모드: 피크별 개별 클립 다운로드 후 샘플 생성

    --download-sections 클립은 타임스탬프가 0부터 시작하므로
    피크 구간을 클립 내 상대 시간으로 변환하여 프레임 추출.

    Args:
        vid: video_id
        title: 영상 제목
        duration: 원본 영상 전체 길이(초)
        peaks: 피크 구간 목록 [{"start_sec": float, "end_sec": float}, ...]
        temp_dir: 임시 파일 저장 디렉토리
        negative_ratio: 네거티브 샘플 비율
        stats: DatasetBuilder._stats 딕셔너리 (직접 수정)
        download_fn: _download_video 코루틴 함수
        extract_fn: _extract_segment_frame 코루틴 함수
        metadata_fn: _seg_metadata 정적 함수
        pick_neg_fn: _pick_negative_segments 정적 함수

    Returns:
        샘플 딕셔너리 리스트
    """

    INSTRUCTION = ("이 게임 영상 프레임을 보고 시청자가 많이 다시 본 하이라이트 구간인지 판단하세요.")
    samples = []
    neg_count = int(len(peaks) * negative_ratio)
    neg_segments = pick_neg_fn(peaks, duration, neg_count)

    # 포지티브 샘플: 피크 구간별 클립 다운로드
    for peak in peaks:
        p_start = peak["start_sec"]
        p_end = peak["end_sec"]
        clip_offset = max(0, p_start - 30)      # 다운로드 시작점 (앞 30초 여유)
        video_path = temp_dir / f"{vid}_{int(p_start)}.mp4"
        try:
            await download_fn(vid, video_path, [peak])
            if not video_path.is_file():
                continue
            rel_start = p_start - clip_offset   # 클립 내 상대 시간
            rel_end = p_end - clip_offset
            frames = await extract_fn(video_path, vid, rel_start, rel_end)
            if not frames:
                continue
            meta = metadata_fn(vid, title, duration, p_start, p_end)
            samples.append({"instruction": INSTRUCTION, "images": frames, "metadata": meta, "output": "하이라이트"})
            stats["positive_samples"] += 1
        except Exception as e:
            logger.warning(f"포지티브 샘플 실패 ({vid} {p_start}~{p_end}): {e}")
        finally:
            if video_path.is_file():
                video_path.unlink()

    # 네거티브 샘플: 비피크 구간별 클립 다운로드
    for seg_s, seg_e in neg_segments:
        clip_offset = max(0, seg_s - 30)
        video_path = temp_dir / f"{vid}_neg_{int(seg_s)}.mp4"
        try:
            await download_fn(vid, video_path, [{"start_sec": seg_s, "end_sec": seg_e}])
            if not video_path.is_file():
                continue
            rel_start = seg_s - clip_offset
            rel_end = seg_e - clip_offset
            frames = await extract_fn(video_path, vid, rel_start, rel_end)
            if not frames:
                continue
            meta = metadata_fn(vid, title, duration, seg_s, seg_e)
            samples.append({"instruction": INSTRUCTION, "images": frames, "metadata": meta, "output": "일반"})
            stats["negative_samples"] += 1
        except Exception as e:
            logger.warning(f"네거티브 샘플 실패 ({vid} {seg_s}~{seg_e}): {e}")
        finally:
            if video_path.is_file():
                video_path.unlink()

    return samples