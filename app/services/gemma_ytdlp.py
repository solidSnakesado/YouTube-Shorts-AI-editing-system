# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: yt-dlp 다운로드 (Gemma 데이터 재구축 - 모듈 C-1). 두 가지 모드:
#       (1) download_video_with_audio: 풀영상 다운로드 (절대 타임스탬프 보존, 풀백용)
#       (2) download_video_section: 피크 구간만 다운로드 (39일차 추가, 풀영상 회피)
# 의존: app.services.dataset_utils.refresh_firefox_cookies (쿠키 갱신 공용 유틸 재사용)
# 39일차: 롱플레이(수 시간) 풀다운로드 병목 해소 위해 구간 다운로드 추가.
#   - 7시간 영상에서 ~2분치 클립만 받도록 -> 다운로드 수십~100배 단축
#   - --download-sections + --force-keyframes-at-cuts: 정확 컷 + 출력 0초 시작 -> 상대 타임스탬프 추출
# 45일차 수정(1회): 두 명령에 --extractor-args "youtube:player_client=web" 추가. 사유: yt-dlp
#   기본 TVHTML5 client의 스트림 URL을 ffmpeg(--download-sections 내부)가 요청하면 403.
#   메타조회(--list-formats)는 되나 구간 다운로드만 실패하던 원인. web client URL은 ffmpeg
#   요청이 쿠키와 함께 통과(동일 영상·구간 단일테스트로 성공 확인). 변경 라인 전달 메시지 참조.

"""Gemma 데이터 재구축용 yt-dlp 다운로드 헬퍼 (풀영상 / 피크 구간 두 모드)"""

import asyncio
from pathlib import Path

from loguru import logger

from app.services.dataset_utils import refresh_firefox_cookies

def _cookie_opts() -> list[str]:
    """Firefox 쿠기 갱신(공용 유틸) 후 yt-dlp --cookies 옵션 반환 (없으면 빈 리스트)"""

    cookie_file = Path("data/youtube_cookies.txt")
    refresh_firefox_cookies(str(cookie_file))       # Firefox 쿠기 자동 갱신 (공용 유틸)
    return ["--cookies", str(cookie_file)] if cookie_file.is_file() else []

async def _run_ytdlp(cmd: list[str], video_id: str, what: str) -> None:
    """yt-dlp 서브프로세스 실행 + 실패 시 RuntimeError (다운로드 함수 공통 헬퍼)"""

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp {what} 실패 (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[:300]}"
        )

async def download_video_with_audio(
    video_id: str,
    output_path: Path,
    socket_timeout: int = 120,
) -> None:
    """yt-dlp로 영상을 오디오 포함 전체 다운로드 (360p).
    --download-sections 미사용 -> 절대 타임스탬프 보존 (피크 구간 정렬용).
    오디오를 포함해 받으므로 1fps 프레임 추출과 오디오 세그먼트 추출 양쪽에 사용.

    Args:
        video_id: YouTube video id
        output_path: 저장 경로 (.mp4, 상위 디렉토리 자동 생성)
        socket_timeout: yt-dlp 소켓 타임아웃 (초)

    Raises:
        RuntimeError: yt-dlp 실행 실패
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        # 39일차: 360p(640x360, 너비 640>=512) + 오디오 -> 512px 프레임 추출 시 업스케일 방지
        #   '18' = YouTube 포맷 ID. 360p MP4에 영상(H.264)+오디오(AAC)가 한 파일로 묶인 결합본
        #       (progressive/muxed). 결합본은 ffmpeg 병합 불필요 (요즘 고화질 DASH는 영상/오디오가 분리
        #       제공되어 받은 뒤 병합해야 함). 18은 거의 모든 영상에 존재해 안정적
        #   포맷 선택 순서: 18(결합본, 병합X) -> 다른 360p 이하 결합 mp4 -> best video(bv)+best audio(ba) (분리->병합) 풀백
        "-f", "18/best[height<=360][ext=mp4]/bestvideo[height<=360]+bestaudio",
        "--merge-output-format", "mp4",             # 분리 스트림(bv+ba) 병합 시 mp4 출력
        "-o", str(output_path), "--no-playlist",
        "--socket-timeout", str(socket_timeout),
        "--no-warnings", "--js-runtimes", "node",   # yt-dlp JS 챌린지 대응 (node 필요)
        # 45일차: web client URL로 받아야 ffmpeg 구간요청 403 회피(TVHTML5 URL은 ffmpeg서 403)
        "--extractor-args", "youtube:player_client=web",
        *_cookie_opts(), url,
    ]
    logger.info(f"Gemma 영상 다운로드 시작 (오디오 포함): {video_id}")
    await _run_ytdlp(cmd, video_id, "다운로드")
    logger.info(f"Gemma 영상 다운로드 완료: {video_id}")

async def download_video_section(
    video_id: str,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    socket_timeout: int = 120,
) -> None:
    """yt-dlp로 영상의 [start_sec, end_sec] 구간만 다운로드 (오디오 포함). 39일차 신규
    
    풀영상 대신 피크 구간만 받아 롱플레이 다운로드 병목을 해소한다.
    - itag 18(progressive) 우선: --download-section가 구간만 fetch(풀 다운로드 안 함)하며
      DASH 스트림은 throttle(!110KiB/s)되나 progressive는 ~700KiB/s로 ~4배 빠름(CLI 실측)
      iteg 18 부재 시에만 H.264 DASH(avc1)로 폴백(AV1 회피 -> 재인코딩 경감).
    - --force-keyframes-at-cuts: 요청 구간을 정확히 잘라 출력 파일이 0초부터 시작
      (키프레임 스냅 불확실성 제거). 가벼운 재인코딩 동반(내용 동일, byte는 상이 가능).
    -> 호출측은 추출 시 절대 시각이 아닌 상대(0-base) 타임스탬프를 사용해야 한다.
    
    Args:
        video_id: YouTube video id
        start_sec: 구간 시작 (영상 절대 시각, 초)
        end_sec: 구간 끝 (영상 절대 시각, 초)
        output_path: 저장 경로 (.mp4, 상위 디렉토리 자동 생성)
        socket_timeout: yt-dlp 소켓 타임 아웃 (초)

    Raises:
        RuntimeError: yt-dlp 실행 실패
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    section = f"*{start_sec}-{end_sec}"     # '*' = 시간 기준 구간 (초)
    cmd = [
        "yt-dlp",
        # 39일차: itag 18(progressive) 우선 -> DASH throttle 회피(구간 ~10초 vs DASH 37~41초, 실측).
        #   18 부재 시 H.264 DASH(avc1)로 폴백(AV1 회피). 640x360(너비>=512)로 512px 추출 업스케일 방지
        "-f", "18/best[height<=360][ext=mp4]/bestvideo[height<=360][vcodec^=avc1]+bestaudio/best[height<=360]",
        "--download-sections", section,         # 구간만 다운로드 (절대 시각 기준)
        "--force-keyframes-at-cuts",            # 정확 컷 -> 출력 0초 시작 (상대 추출 가능)
        "--merge-output-format", "mp4",
        "-o", str(output_path), "--no-playlist",
        "--socket-timeout", str(socket_timeout),
        "--no-warnings", "--js-runtimes", "node",
        # 45일차: web client URL로 받아야 ffmpeg 구간요청 403 회피(핵심 - 구간 다운로드 실패 원인)
        "--extractor-args", "youtube:player_client=web",
        *_cookie_opts(), url,
    ]
    logger.info(f"Gemma 구간 다운로드 시작: {video_id} | {start_sec:.0f}~{end_sec:.0f}초")
    await _run_ytdlp(cmd, video_id, "구간 다운로드")
    logger.info(f"Gemma 구간 다운로드 완료: {video_id} | {start_sec:.0f}~{end_sec:.0f}초")