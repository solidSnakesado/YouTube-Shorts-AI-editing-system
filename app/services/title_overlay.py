# 계층: 비즈니스 로직 계층 (Service)
# 역할: LLM 생성 제목(title_suggestion)을 영상 상단에 FFmpeg drawtext로 오버레이
# 분리 근거: editing_service.py 300중 한계 대응 (25일차 - 300줄 한계 대응)
# 의존: FFmpeg (drawtext 필터), config (폰트/경로 설정)
# 26일차 신규

"""제목 오버레이 - FFmpeg drawtext로 title_suggestion을 영상 상단에 삽입"""

import asyncio
import re
from pathlib import Path

from loguru import logger

from app.core.config import settings

# --- 오버레이 스타일 상수 ---
FONT_NAME   = "Noto Sans CJK KR"    # 한글 지원 폰트 (verify_font로 사전 확인)
FONT_SIZE   = 56                    # 제목 폰트 크가 (1080x1920 기준)
FONT_COLOR  = "white"               # 제목 텍스트 색상
BOX_COLOR   = "black@0.55"          # 반투명 배경 박스 색상
BOX_BORDER  = 18                    # 배경 박스 패딩 (픽셀)
MARGIN_TOP  = 60                    # 상단 여백 (픽셀)
DISPLAY_SEC = 4.0                   # 제목 표시 시간 (초), 0이면 영상 정체

# --------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------

async def apply_title_overlay(input_path: str, output_path: str, title: str, diaplay_sec: float = DISPLAY_SEC) -> bool:
    """
    영상 상단에 제목 텍스트를 오버레이한다.

    FFmpeg drawtext 필터 사용:
    - 반투명 검정 박스 배경 + 흰색 굵은 텍스트
    - display_sec 동안 표시 후 사라짐 (0이면 영상 전체 표시)
    - 한글 포함 15자 이내 제목 기준 설계

    Args:
        input_path:     입력 영상 경로
        output_path:    출력 영상 경로
        title:          표시할 제목 문자열 (title_suggestion)
        display_sec:    제목 표시 지속 기간 (초). 0이면 전체

    Returns:
        True(성공) / False(실패)
    """

    if not title or not title.strip():
        logger.warning("제목 오버레이: 제목이 비어 있어 건너뜀")
        return False
    
    sanitized = _sanitize_drawtext(title.strip())
    font_path = _resolve_font_path
    vf_filter = _build_drawtext_filter(sanitized, font_path, diaplay_sec)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf_filter,
        "-c:v", "h264_nvenc",
        "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ),
        "-c:a", "copy",
        output_path,
    ]

    logger.info(f"제목 오버레이 시작 | '{title}' | {Path(input_path).name}")

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(f"제목 오버레이 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"제목 오버레이 완료: {output_path}")
    return True

# --------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------

def _sanitize_drawtext(text: str) -> str:
    """
    drawtext 필터에서 특수문자로 인식되는 문자를 이스케이프.
    FFmpeg drawtext는 : ' \ 를 특수 처리한다.
    """

    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    # 이모지 등 제어문자 제거 (렌더링 오류 방비)
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    return text

def _resolve_font_path() -> str:
    """
    시스템에 설치된 Noto Sans CJK KR 폰트 경로를 반환
    미설치 시 빈 문자열 반환 -> drawtext가 기본 폰트로 폴백
    """

    import subprocess
    try:
        result = subprocess.run(["fc-list", "Noto Sans CJK KR", "--format=%{file}\n"], capture_output=True, text=True, timeout=5)
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if lines:
            # Regular 또는 Medium 굵기 우선
            for line in lines:
                if "Regular" in line or "Medium" in line:
                    return line
            return lines[0]
    except Exception as e:
        logger.warning(f"폰트 경로 조회 실패: {e}")
    return ""

def _build_drawtext_filter(text: str, font_path: str, diaplay_sec: float) -> str:
    """
    FFmpeg drawtext vf 필터 문자열 생성.

    레이아웃:
        - x: 중앙 정렬 (w-text_w)/2
        - y: 상단 MARGIN_TOP 픽셀
        - 반투명 박스 배경으로 가독성 확보
        - display_sec > 0 이면 해당 시간 이후 alpha=0 (숨김)
    """

    font_opt = f":fontfile='{font_path}'" if font_path else ""

    # 표시 시간 제한 (alpha 채널 활용)
    if diaplay_sec > 0:
        enable = f":enable='between(t,0,{diaplay_sec})'"
    else:
        enable = ""

    return (
        f"drawtext=text='{text}'"
        f"{font_opt}"
        f":fontsize={FONT_SIZE}"
        f":fontcolor={FONT_COLOR}"
        f":box=1:boxcolor={BOX_COLOR}:boxborderw={BOX_BORDER}"
        f":x=(w-text_w)/2"
        f":y={MARGIN_TOP}"
        f"{enable}"
    )