# 계층: 비즈니스 로직 계층 (Service)
# 역할: 쇼츠 최종 출력 파일명 생성 (sanitize + 충돌 방지 + 풀백 제목)
# 분리 출처: editing_service.py (25일차 - 300줄 한계 대응)
# 의존: domain.Short(타입 힌트 전용), pathlib

"""쇼츠 출력 파일명 빌더 - sanitize, 충돌 순번, 풀백 제목"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.models.domain import Shorts

# --------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------

def build_final_path(short: "Shorts", output_dir: Path) -> Path:
    """
    쇼츠 최종 출력 경로를 결정한다.
    
    우선순위:
        1. title_suggestion (한글 포함 시)
        2. highlight_reason 앞 20자 + 타임코드
        3. 타임코드만 (clip_{start}s_{end}s)

    충돌 방지: 동일 파일명 존재 시 _01, _02 순번 suffix 추가

    Args:
        short: Shorts 엔티티 (title_suggestion, highlight_reason, start_sec, end_sec 사용)
        output_dir: 출력 디렉터리 (settings.output_path)
    
    Returns:
        충돌 없는 최종 .mp4 경로
    """

    stem = _pick_stem(short)
    return _resolve_collision(output_dir, stem)

# --------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------

def _sanitize(name: str) -> str:
    """
    파일명 불가 문자 제거 + 80자 제한
    한글이 하나도 없으면 빈 문자열 반환 (호출부에서 풀백 처리)
    """

    if not any('\uAC00' <= c <= '\uD7A3' for c in (name or "")):
        return ""

    safe = re.sub(r'[\\/*?:"<>|]', '', name)
    safe = safe.replace('\n', ' ').replace('\r', '').strip()
    safe = re.sub(r'\s+', '_', safe)
    return safe[:80]

def _timecode(start: float, end: float) -> str:
    """타임코드 문자열 생성 ()"""

    def fmt(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}m{s:02d}s"
    return f"{fmt(start)}_{fmt(end)}"

def _pick_stem(short: "Shorts") -> str:
    """
    파일명 줄기(stem) 결정
    title_suggestion -> highlight_reason + 타임코드 -> 타임코드 순으로 폴백
    """

    start = getattr(short, "start_sec", 0) or 0
    end = getattr(short, "end_sec", 0) or 0
    tc = _timecode(start, end)

    # 1순위: title_suggestion (한글 포함)
    title = getattr(short, "title_suggestion", "") or ""
    stem = _sanitize(title)
    if stem:
        logger.debug(f"파일명 stem: title_suggestion 사용 -> {stem}")
        return stem

    # 2순위: highlight_reason 앞 20자 + 타임코드
    reason = getattr(short, "highlight_reason", "") or ""
    reason_clean = _sanitize(reason[:30])
    if reason_clean:
        stem = f"{reason_clean}_{tc}"
        logger.debug(f"파일명 stem: highlight_reason + 타임코드 -> {stem}")
        return stem

    # 3순위: 타임코드만
    stem = f"clip_{tc}"
    logger.debug(f"파일명 stem: 타임코드 전용 -> {stem}")
    return stem

def _resolve_collision(output_dir: Path, stem: str, max_suffix: int = 99) -> Path:
    """
    충돌 방지: {stem}.mp4 존재 시 {stem}_01.mp4, _02.mp4 ... 순번 추가
    max_suffix 초과 시 마지막 순번을 덮어씀 (극단적 상황 대비)
    """

    candidate = output_dir / f"{stem}.mp4"
    if not candidate.exists():
        return candidate

    for i in range(1, max_suffix + 1):
        candidate = output_dir / f"{stem}_{i:02d}.mp4"
        if not candidate.exists():
            logger.debug(f"파일명 충돌 -> {candidate.name} 으로 변경")
            return candidate

    # 모든 순번 소진 시 마지막 경로 반환 (덮어쓰기)
    logger.warning(f"파일명 순번 소진({max_suffix}), 마지막 경로 사용: {candidate.name}")
    return candidate