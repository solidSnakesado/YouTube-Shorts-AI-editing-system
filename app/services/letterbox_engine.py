# 계층: 비즈니스 로직 계층 (Service)
# 역할: 블러 배경 레터박스 레이아웃 FFmpeg 실행
# 분리 근거: reframe_engine.py 300줄 한계 대응 (26일차 신규)
# 의존: config (NVENC 설정), reframe_engine.TARGET_RESOLUTIONS (해상도 상수)
# 흐름: editing_service.reframe_clip(layout="letterbox") -> run_ffmpeg_letterbox()

"""레터박스 엔진 - 블러 배경 위에 원본 비율 영상을 중앙 배치"""

import asyncio
from pathlib import Path

from loguru import logger
from app.core.config import settings
from app.services.reframe_engine import TARGET_RESOLUTIONS

# 블러 강도 (sigma): 높을 수록 더 강한 블러
BLUR_SIGMA = 30

# --------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------

async def run_ffmpeg_letterbox(source_path: str, output_path: str, aspect_ratio: str) -> bool:
    """
    레터박스 레이아웃: 원본 영상 비율을 유지하면서 타겟 해상도에 맞게 배치.
    남는 공간은 원본 영상을 블러 처리한 배경으로 채워 검정 패딩보다 시각적으로 자연스럽다.

    FFmpeg filtergraph 구조:
        입력 -> split(2)
            -> [fg]: scale + pad (원본 비율 유지, 중앙 배치, 투명 패딩)
            -> [bg]: scale + crop + gblur (배경 블러)
        overlay: bg 위에 fg를 중앙 합성

    Args:
        source_path:    입력 클립 경로 (편집 전 영상)
        output_path:    출력 경로
        aspect_ratio:   타겟 비율 문자열 (예: "9:16", "16:9")
                        TARGET_RESOLUTIONS에 정의된 값이어야 함
    
    Returns:
        True(성공) / False(실패 - 로그에 FFmpeg stderr 기록)
    """

    if aspect_ratio not in TARGET_RESOLUTIONS:
        logger.error(f"레터박스: 미지원 종횡비 {aspect_ratio} | 지원: {list(TARGET_RESOLUTIONS)}")
        return False
    
    tw, th = TARGET_RESOLUTIONS[aspect_ratio]
    vf = _build_letterbox_filter(tw, th)

    cmd = [
        "ffmpeg", "-y", "-hwaccel", settings.FFMPEG_HWACCEL,
        "-i", source_path,
        "-filter_complex", vf,
        "-c:v", "h264_nvenc", "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ),
        "-c:a", "copy",
        output_path,
    ]

    logger.info(f"레터박스 시작 | {aspect_ratio} ({tw}x{th}) | {Path(source_path).name}")

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"레터박스 FFmpeg 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"레터박스 완료: {output_path}")
    return True

# --------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------

def _build_letterbox_filter(tw: int, th: int) -> str:
    """
    FFmpeg filter_complex 문자열 생성

    전경(fg):   원본 비율 유지 + 타겟 캔버스에 중앙 배치 (투명 패딩)
    배경(bg):   타겟 해상도로 fill 후 블러 적용
    최종:       bg 위에 fg를 중앙 overlay

    Args:
        tw: 타겟 너비 (픽셀)
        th: 타겟 높이 (픽셀)

    Returns:
        filter_complex 문자열
    """

    # 전경: 비율 유지 -> 패딩 (black@0 = 투명, overlay에서 bg가 비침)
    fg = (
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black@0"
    )

    # 배경: 비율 무시 fill -> 중앙 크롭 -> 블러
    bg = (
        f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
        f"crop={tw}:{th},"
        f"gblur=sigma={BLUR_SIGMA}"
    )

    return (
        f"[0:v]split=2[fg_in][bg_in];"
        f"[bg_in]{bg}[bg];"
        f"[fg_in]{fg}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )