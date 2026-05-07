# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 전사 데이터를 시간 기반 청크로 분할 + 청크별 하이라이트 후보 병합/재랭킹
#       analysis_service.py의 300줄 규칙으로 분리된 모듈 (17일차 신규)
# 의존: 없음 (전사 dict와 하이라이트 dict 리스트를 인자로 받아 처리하는 순수 함수)
# MVA 원칙: 인프라 책임 없음 - 순수 데이터 변환 로직만 포함
#
# 흐름:
#   1. analysis_service._run_highlight_extraction 이 split_transcript_into_chunks 호출
#   2. 각 청크에 대하 llm_highlight_extractor.build_highlight_prompt -> call_llm -> parse_highlights
#   3. 모든 청크 결과를 merge_and_rerank_highlights 로 병합하여 최종 하이라이트 확정
#
# 배경 (16일차 발견된 이슈):
#   - 93분 영상 처리 시 프롬프트가 약 47K 토큰으로 부풀어 LLM_CTX_SIZE=8192 초과
#   - 컨텍스트를 48K로 올리면 KV 캐시 VRAM 스필오버 -> PCIe 병목으로 실행 결과 75분 이상
#   - 해결: 전사를 10분 청크로 분할 -> 각 청크당 프롬프트 약 5K -> 8K 컨텍스트 안전

"""
전사 청크 분할기 - 장편 영상을 시간 기반 청크로 분할, 청크별 하이라이트 재랭킹

핵심 원리:
    - 각 LLM 호출릐 프롬프트는 10분치 전사(약 5K 토큰)로 제한
    - 청크 경계의 발화 유실 방지를 위해 30초 오버랩 적용
    - 여러 청크에서 중복 선정된 하이라이트는 IoU 기반으로 겹침 제거
"""

from loguru import logger

# 마지막 청크가 너무 짧으면 이전 청크와 병합하는 기준 (초)
_MIN_TAIL_CHUNK_SEC = 60.0

# --------------------------------------------------------------
# 1. 청크 분할
# --------------------------------------------------------------
def split_transcript_into_chunks(transcript: dict, chunk_duration_sec: float = 600.0, 
                                 overlap_sec: float = 30.0) -> list[dict]:
    """
    전사 데이터를 시간 기반 청크로 분할

    각 청크는 원본 transcript 구조를 모방하여 segments, duration_sec, language를 가짐
    세그먼크의 start/end는 원본 영상 기준 절대 시각을 그대로 유지
    (LLM 응답이 절대 시각으로 반환되도록 유도하기 위함)

    Args:
        transcript: Whisper 전사 결과 dict (segments 포함)
        chunk_duration_sec: 청크 기본 길이 (초), 기본 600초(10분)
        overlap_sec: 청크 간 오버랩 (초), 기본 30초 (앞뒤 각 15초)

    Returns:
        list of chunk dict:
            {
                "chunk_index": int,                 # 0부터 시작
                "start_offset_sec": float,          # 원본 영상 기준 청크 시작
                "end_offset_sec": float,            # 원본 영상 기준 청크 끝
                "duration_sec": float,              # 청크 길이 (LLM 프롬프트용)
                "language": str,                    # 원본 transcript의 language 전파
                "segments": list[dict],             # 해당 시간대의 세그먼트 (절대 시각 유지)
            }
        짧은 영상(total <= chunk_duration_sec)은 청크 1개 반환 (오버랩 없음)
    """
    
    total_duration = float(transcript.get("duration_sec", 0) or 0)
    language = transcript.get("language", "unknown")
    segments = transcript.get("segments", [])

    # 엣지 케이스: 영상이 없거나 청크 길이 이하인 경우 -> 청크 1개 (오버랩 없음)
    if total_duration <= 0 or total_duration <= chunk_duration_sec:
        logger.info(f"청크 분할 불필요: duration={total_duration:.1f}s -> 1개 청크")
        return [_build_chunk(0, 0.0, total_duration, language, segments)]
    
    boundaries = _compute_chunk_boundaries(total_duration, chunk_duration_sec, overlap_sec)
    chunks = []
    for idx, (start, end) in enumerate(boundaries):
        chunk_segments = _filter_segments_by_range(segments, start, end)
        chunks.append(_build_chunk(idx, start, end, language, chunk_segments))

    logger.info(
        f"청크 분할 완료: total={total_duration:.1f}s, chunks={len(chunks)}, "
        f"chunk_size={chunk_duration_sec}s, overlap={overlap_sec}s"
    )

    return chunks

def _compute_chunk_boundaries(total_duration: float, chunk_duration_sec: float, 
                              overlap_sec: float) -> list[tuple[float, float]]:
    """
    청크 경계 (start, end) 튜플 목록 계산
    기본 전진 폭 = chunk_duration_sec, 오버랩 = overlap_sec (앞뒤 각 overlap_sec / 2)
    마지막 청크가 _MIN_TAIL_CHUNK_SEC 미만이면 이전 청크와 병합
    """
    
    half_overlap = overlap_sec / 2.0
    boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < total_duration:
        raw_start = cursor
        raw_end = cursor + chunk_duration_sec

        # 앞쪽 오버랩: 첫 청크는 0초, 이후는 앞쪽으로 half_overlap 확장
        start = max(0.0, raw_start - half_overlap) if boundaries else 0.0

        # 뒤쪽 오버랩: 마지막 청크는 total, 이후는 뒤쪽으로 half_overlap 확장
        end = min(total_duration, raw_end + half_overlap)
        boundaries.append((start, end))

        cursor += chunk_duration_sec
        # 잔여 구간이 최소 길이 미만이면 이전 청크를 확장하여 흡수
        if cursor < total_duration and (total_duration - cursor) < _MIN_TAIL_CHUNK_SEC:
            prev_start, _ = boundaries[-1]
            boundaries[-1] = (prev_start, total_duration)
            break

    return boundaries

def _filter_segments_by_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """
    세그먼트 중 [start, end] 범위와 교집합이 있는 것을 선택 (절대 시각 유지)
    """
    
    filtered = []
    for seg in segments:
        seg_start = float(seg.get("start", 0) or 0)
        seg_end = float(seg.get("end", 0) or 0)
        if seg_end <= start or seg_start >= end:
            continue
        filtered.append(seg)

    return filtered

def _build_chunk(idx: int, start: float, end: float, language: str, segments: list[dict]) -> dict:
    """
    청크 dict 구성 핼처 (원본 transcript 구조를 모방)
    """
    
    return {
        "chunk_index": idx,
        "start_offset_sec": round(start, 3),
        "end_offset_sec": round(end, 3),
        "duration_sec": round(end - start, 3),
        "language": language,
        "segments": segments,
    }

# --------------------------------------------------------------
# 2. 병합 + 재랭킹
# --------------------------------------------------------------
def merge_and_rerank_highlights(chunk_results: list[list[dict]], max_shorts: int, 
                                iou_threshold: float = 0.3) -> list[dict]:
    """
    여러 청크에서 추출된 하이라이트 후보를 병합, 시간 겹침 제거 후 재랭킹

    알고리즘:
        1. 모든 청크 후보를 flat list 로 합침
        2. hook_score 내림차순 정렬
        3. 빈 결과 리스트에 순서대로 추가하되, 기존 항목과 IoU >= threshold 면 스킵
        4. max_shorts 개 도달 시 중단

    Args:
        chunk_results: 각 청크의 하이라이트 리스트 (parse_highlights 결과들을 리스트)
        max_shorts: 최종 선정할 쇼츠 수
        iou_threshold: 시간 겹침 판정 임계값 (0.3 = 30% 이상 겹치면 동일 하이라이트)

    Returns:
        최대 max_shorts 개의 하이라이트 dict (hook_score 내림차순)
    """
    
    # 모든 청크 결과를 flat list 로
    all_candidates: list[dict] = []
    for chunk_list in chunk_results:
        if chunk_list:
            all_candidates.extend(chunk_list)

    if not all_candidates:
        logger.warning("재랭킹: 모든 청크가 비어 있음")
        return []
    
    # hook_score 내림차순 정렬 (없으면 0 취급)
    all_candidates.sort(key=lambda h: h.get("hook_score", 0), reverse=True)

    selected: list[dict] = []
    for cand in all_candidates:
        is_duplicate = any(_compute_time_iou(cand, prev) >= iou_threshold for prev in selected)
        if is_duplicate:
            continue

        selected.append(cand)
        if len(selected) >= max_shorts:
            break

    logger.info(
        f"재랭킹 완료: 후보 {len(all_candidates)}개 -> 선정 {len(selected)}개 "
        f"(max={max_shorts}, iou_threshold={iou_threshold})"
    )
    
    return selected

def _compute_time_iou(a: dict, b: dict) -> float:
    """
    두 하이라이트의 시간 IoU (Intersection over Union) 계산
    IoU = 겹치는 시간 / 합집합 시간, 반환값: [0.0, 1.0]
    """
    
    try:
        a_start = float(a.get("start_sec", 0) or 0)
        a_end = float(a.get("end_sec", 0) or 0)
        b_start = float(b.get("start_sec", 0) or 0)
        b_end = float(b.get("end_sec", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0.0, inter_end - inter_start)
    if intersection <= 0:
        return 0.0
    
    union = (a_end - a_start) + (b_end - b_start) - intersection
    if union <= 0:
        return 0.0
    
    return intersection / union