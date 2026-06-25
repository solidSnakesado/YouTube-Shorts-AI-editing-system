# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 영상의 특정 구간을 Gemma 오디오 입력용 WAV(16kHz mono)로 추출
#       Gemma 4 E4B 오디오 피벗 데이터 재구축 - 모듈 A (frame_extractor의 오디오 대응)
# 의존: app.core.gemma_config (GEMMA_AUDIO_MAX_SEC)
# 39일차 신규: [1fps 프레임 + segment 오디오] 데이터셋의 오디오 절반 담당
#   - 기존 frame_extractor와 동일한 async ffmpeg 스타일, Qwen 스택 무수정 (격리)

"""Gemma 오디오 세그먼트 추출기 - 영상 구간 -> 16kHz mono WAV (30s 캡)"""

import asyncio
from pathlib import Path

from loguru import logger

from app.core.gemma_config import gemma_settings

async def _has_audio_stream(video_path: Path) -> bool:
    """39일차: ffprobe로 오디오 스트림 존재 여부 확인 (무음 게임플레이 등 스킵용)"""
    
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a",             # 오디오 스트림만 선택
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return bool(stdout.decode().strip())

async def extract_audio_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    save_path: Path,
    max_sec: int | None = None,
) -> Path | None:
    """영상의 [start_sec, end_sec] 구간을 16kHz momo WAV로 추출
    
    Gemma 4 오디오 입력 규격(16kHz mono)에 맞춰 추출하며, 구간 길이를
    max_sec(기본 GEMMA_AUDIO_MAX_SEC=30s)로 지정, frame_extractor와 동일하게
    -ss를 입력 전에 두어 빠른 seek + 프래임 추출과 동일 기준으로 정렬한다.
    
    Args:
        video_path: 소스 영상 (오디오 트랙 포함)
        start_sec: 구간 시작 (초)
        end_sec: 구간 종료 (초)
        save_path: 출력 WAV 경로 (상위 디렉토리 자동 생성)
        max_sec: 오디오 길이 상한 (기본 gemma_settings.GEMMA_AUDIO_MAX_SEC)

    Returns: 
        성공 시 저장 경로 (Path), 오디오 트랙 없음/구간 무효/무음 시 None

    Raises:
        FileNotFoundError: 영상 파일 미존재
        RuntimeError: ffmpeg 실행 실패
    """

    video_path = Path(video_path)
    if not  video_path.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    
    seg_dur = end_sec - start_sec
    if seg_dur <= 0:
        logger.warning(f"오디오 구간 무효: {start_sec:.1f}~{end_sec:.1f}초")
        return None

    # 39일차: 오디오 트랙 없는 영상(무음 게임플레이 등)은 스킵 -> None
    if not await _has_audio_stream(video_path):
        logger.info(f"오디오 트랙 없음 -> 스킵: {video_path.name}")
        return None
    
    cap = max_sec or gemma_settings.GEMMA_AUDIO_MAX_SEC
    dur = min(seg_dur, cap)         # 30s 캡

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])     # 입력 전 seek (빠름, 프레임과 동일 기준)
    cmd.extend([
        "-i", str(video_path),
        "-t", str(dur),
        "-vn",                                  # 비디오 제거 (오디오만)
        "-ac", "1",                             # mono
        "-ar", "16000",                         # 16kHz (Gemma 오디오 규격)
        str(save_path),                         
    ])

    logger.info(
        f"오디오 추출 시작 | {video_path.name} | "
        f"{start_sec:.0f}~{start_sec + dur:.0f}초 ({dur:.0f}s, 캡 {cap}s) -> {save_path.name}"
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 오디오 추출 실패: {stderr.decode(errors='replace')[:500]}")
    
    # 무음/빈 추출 방어: WAV 헤더만 있는 수준이면 스킵 (16kHz mono 16bit = 32KB/s)
    if not save_path.is_file() or save_path.stat().st_size < 1000:
        logger.warning(f"추출된 오디오가 비어잇음 -> 스킵: {save_path}")
        return None
    
    logger.info(f"오디오 추출 완료 | {save_path} ({save_path.stat().st_size // 1024}KB)")
    return save_path