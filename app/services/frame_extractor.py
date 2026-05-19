# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 영상에서 일정 간격으로 프레임을 추출하고 base64로 인코딩
#       VLM(Vision-Language Model) 멀티모달 입력 데이터 생성 전담
# 의존: app.core.config (프레임 추출 설정: 간격, 최대 수, 해상도)
# MVA 원칙: 프레임 추출은 비즈니스 헬퍼, GPU/모델 관리는 인프라에 위임
#
# 사용처:
#   - AnalysisService: VLM 분석 시 영상 프레임 + 전사 텍스트 통합 (15일차 예정)
#
# 14일차 신규:
#   - extract_frames(): FFmpeg로 영상에서 N초 간격 프레임 추출
#   - _encode_frame_to_base64(): JPEG 파일 -> base64 문자열 반환
#   - _get_video_duration(): FFprobe로 영상 길이 조회
#
# 21일차 변경:
#   - extract_frames()에 start_sec/end_sec/save_dir 파라미터 추가 (하위 호환)
#   - save_dir 지정 시 JPEG 파일을 해당 경로에 저장하고 파일 경로 목록 반환
#   - 파인 튜닝 데이터 준비(DatasetBuilder)에서 특정 시간 범위 프레임 추출에 사용
#
# Gemma 4 비주얼 토큰 예산 (설정 가이드):
#   560px -> 프레임당 ~280토큰 -> 20프레임 = ~5,600토큰 (일반 분석)
#   1120px -> 프레임당 ~1,120토큰 -> 10프레임 = ~11,200토큰 (OCR/세부)

"""
프레임 추출기 - 영상에서 VLM 입력용 프레임을 추출하고 base64로 인코딩

FFmpeg를 사용하여 영상에서 일정 간격으로 프레임을 JPEG로 추출한 뒤,
각 프레임을 base64 문자열로 변환하여 VLM API에 이미지로 전달할 수 있게 준비
"""

import asyncio                              # FFmpeg/FFprobe를 비동기 서브프로세스로 실행
import base64                               # JPEG -> base64 문자열 반환
import shutil                               # 임시 프레임 디렉토리 정리
from pathlib import Path                    # 파일/디렉토리 경로 처리

from loguru import logger                   # 구조화된 로깅

from app.core.config import settings

async def extract_frames(
    video_path: Path,
    interval_sec: float | None = None,
    max_frames: int | None = None,
    resolution: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    save_dir: Path | None = None,
) -> list[dict]:
    """
    영상에서 일정 간격으로 프레임을 추출하고 base64로 인코딩

    FFmpeg의 fps 필터로 interval_sec 간격마다 프레임을 JPEG로 저장한 뒤,
    각 파일을 base64 문자열로 반환하여 VLM API 입력에 사용할 수 있는 형태로 반환

    Args:
        video_path: 소스 영상 파일 경로
        interval_sec: 프레임 추출 간격 (초, 기본: settings.FRAME_EXTRACT_INTERVAL_SEC)
        max_frames: 최대 추출 프레임 수 (기본: settings.FRAME_EXTRACT_MAX_FRAMES)
        resolution: 프레임 해상도 - 짧은 변 기준 (기본: settings.FRAME_EXTRACT_RESOLUTION)
        start_sec: 추출 시간 시간 (초, 기본: 0.0) - 21일차 추가
        end_sec: 추출 종료 시간 (초, None이면 영상 끝까지) - 21일차 추가
        save_dir: JPEG 저장 경로 (None이면 base64 반환, 경로 지정 시 파일 저장) - 21일차 추가

    Returns:
        save_dir=None (기본, 하위 호환):
            [{"timestamp_sec": 0.0, "base64": "...", "mine_type": "image/jpeg"}, ...]
        save_dir 지정 시:
            [{"timestamp_sec": 0.0, "path": "/abs/path/frame.jpg", "mine_type": "image/jpeg"}, ...]

    Raises:
        FileNotFoundError: 영상 파일 미존재
        RuntimeError: FFmpeg 실행 실패
    """

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    
    # 설정 기본값 적용
    interval_sec = interval_sec or settings.FRAME_EXTRACT_INTERVAL_SEC
    max_frames = max_frames or settings.FRAME_EXTRACT_MAX_FRAMES
    resolution = resolution or settings.FRAME_EXTRACT_RESOLUTION

    # 영상 길이 조회 -> 실제 필요한 프레임 수 계산
    duration = await _get_video_duration(video_path)
    effective_end = min(end_sec, duration) if end_sec else duration
    effective_duration = max(0.0, effective_end - start_sec)

    # 범위 내 프레임 수 계산
    total_possible = int(effective_duration / interval_sec) + 1
    frame_count = min(total_possible, max_frames)

    # 영상이 짧아서 추출할 프레임이 없는 경우
    if frame_count <= 0:
        logger.warning(f"영상이 너무 짧아 프레임 추출 불가: {effective_duration:.1f}초")
        return []
    
    # 프레임 저장 디렉토리 결정
    # save_dir 지정 시: 해당 경로에 영구 저장 (파인튜닝용)
    # save_dir 미지정: 임시 디렉토리에 저장 후 base64 변환 (기존 동작)
    if save_dir:
        frames_dir = Path(save_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        frames_dir = video_path.parent / f"frames_{video_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        cleanup = True

    try:
        # FFmpeg 명령 구성
        # -ss: 시작 시간 (입력 전 탐색으로 빠른 seek)
        # -t: 추출 구간 길이
        fps_filter = f"fps=1/{interval_sec}"
        scale_filter = f"scale={resolution}:-1"
        output_pattern = str(frames_dir / "frame_%04d.jpg")

        cmd = ["ffmpeg", "-y"]

        # 시작 시간이 0이 아니면 -ss로 탐색 (입력 전 seek)
        if start_sec > 0:
            cmd.extend(["-ss", str(start_sec)])

        cmd.extend(["-i", str(video_path)])

        # 종료 시간이 지정되면 -t로 구간 길이 제한
        if effective_duration < duration:
            cmd.extend(["-t", str(effective_duration)])

        cmd.extend([
            "-vf", f"{fps_filter},{scale_filter}",
            "-frames:v", str(frame_count),
            "-q:v", "3",
            output_pattern,
        ])

        logger.info(
            f"프레임 추출 시작 | {video_path.name} | "
            f"범위: {start_sec:.0f}~{effective_end:.0f}초 | "
            f"간격: {interval_sec}초 | 최대: {frame_count}프레임 | "
            f"해상도: {resolution}px"
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg 프레임 추출 실패: {stderr.decode()[:500]}")
        
        # 추출된 프레임 파일 수집 (파일명 순서 = 시간 순서)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            logger.warning("FFmpeg 성공했지만 추출된 프레임이 없습니다")
            return []
        
        # 결과 구성: save_dir 여부에 따라 base64 또는 파일 경로 반환
        results = []
        for idx, frame_file in enumerate(frame_files):
            timestamp = idx * interval_sec
            if save_dir:
                results.append({
                    "timestamp_sec": round(timestamp, 1),
                    "path": str(frame_file.resolve()),
                    "mime_type": "image/jpeg",
                })
            else:
                b64_str = _encode_frame_to_base64(frame_file)
                results.append({
                    "timestamp_sec": round(timestamp, 1),
                    "base64": b64_str,
                    "mime_type": "image/jpeg",
                })

        logger.info(
            f"프레임 추출 완료 | {len(results)}프레임 | "
            f"{results[0]['timestamp_sec']}~{results[-1]['timestamp_sec']}초"
        )

        return results
    finally:
        # 임시 프레임만 디렉토리 정리 (save_dir 지정 시 보존)
        if cleanup and frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)


def _encode_frame_to_base64(frame_path: Path) -> str:
    """
    JPEG 프레임 파일을 base64 문자열로 변환

    VLM API(OpenAI 호환)에서 이미지를 전달할 때 base64 인코딩 사용.
    data URI 프리픽스는 포함하지 않음 (API 호출 시 별도 조합)

    Args:
        frame_path: JPEG 파일 경로

    Returns:
        base64 인코딩된 문자열 (프리픽스 없음)
    """
    
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def _get_video_duration(video_path: Path) -> float:
    """
    FFprobe로 영상 길이(초) 조회

    FFprobe의 format.duration 필드를 사용하여 빠르게 조회
    별도의 디코딩 없이 컨테이너 메타데이터만 읽으므로 거의 즉시 완료

    Args:
        video_path: 영상 파일 경로

    Returns:
        영상 길이(초) - float

    Raises:
        RuntimeError: FFprobe 실행 실패 또는 duration 파싱 불가
    """
    
    cmd = [
        "ffprobe",
        "-v", "quiet",                                       # 불필요한 로그 제거
        "-show_entries", "format=duration",                 # duration 필드만 출력
        "-of", "default=noprint_wrappers=1:nokey=1",        # 값만 출력 (키 제거)
        str(video_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFprobe 실행 실패: {stderr.decode()[:300]}")
    
    try:
        duration = float(stdout.decode().strip())
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"영상 길이 파싱 실패: stdout='{stdout.decode().strip()}' | {e}")
    
    logger.debug(f"영상 길이 조회: {video_path.name} -> {duration:.1f}초")
    return duration