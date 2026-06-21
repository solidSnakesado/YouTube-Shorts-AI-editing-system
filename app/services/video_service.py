# 계층: 비즈니스 로직 계층 (Service)
# 역할: 유투브 영상 다운로드와 전처리 비즈니스 로직을 담당
#       DB 접근은 반드시 레포지토리를 통해서만 수행 (직접 SQL 금지)
#       HTTP 요청/응답 객체를 직접 다루지 않는다 (프레임워크 독립)
# 의존: ProjectRepository (DI로 주입 받음)
# MVA 원칙: 서비스 계층은 순수 비즈니스 로직만 포함, DB도 HTTP도 알지 못함

"""
비디오 서비스

유투브 영상 다운로드 및 전처리 비즈니스 로직을 담당
"""

import asyncio                  # 비동기 서브르포세스 실행 (yt-dlp, ffmpeg)
import json                     # ffprobe 출력 JSON 파싱
from pathlib import Path
from typing import Optional

from loguru import logger       # loguru: 구조화된 로깅 라이브러리 (print 대신 사용)

from app.core.config import settings
from app.models.domain import Project, ProjectStatus
from app.repositories.project_repository import ProjectRepository

class VideoService:
    """
    비디오 서비스

    책임:
        1. 프로젝트(영상 분석 작업) 생성/조회
        2. yt-dlp로 유투브 영상 다운로드
        3. FFmpeg로 오디오 추출 (Whisper 입력용)
        4. FFprobe로 영상 메타데이터 추출
    """
    
    def __init__(self, project_repo: ProjectRepository):
        """
        DI로 레포지토리를 주입 받음
        VideoService는 ProjectRepository의 메서드만 호출, 
        AsyncSession 이나 SQL 쿼리를 직접 알지 못함
        """
        
        self.project_repo = project_repo

    async def create_project(self, youtube_url: str) -> Project:
        """
        새 프로젝트 생성

        Project 엔티티를 만들고 DB에 저장
        초기 상태는 PENDING (아직 다운로드 시작 안 됨)
        """
        
        project = Project(youtube_url=str(youtube_url))
        project = await self.project_repo.create(project)                   # BaseRepository.create() 호출
        logger.info(f"프로젝트 생성: {project.id} | URL: {youtube_url}")

        return project
    
    async def download_video(self, project_id: str, quality: int = 1080) -> Optional[Project]:
        """
        유투브 영상 다운로드 및 오디오 추출.

        Args:
            quality: 다운로드 최대 해상도(height). 기본 1080(발행용)
                     38일차: 라벨링 단계는 480 전달 - 프레임이 336px로 추출되므로
                     화질/학습 무영향, 다운로드량만 ~5배 절감   

        전체 흐름:
            1. 상태를 DOWNLOADING으로 변경
            2. yt-dlp로 영상 다운로드 (1080p MP4)
            3. FFmpeg로 오디오 추출 (16kHz mono WAV)
            4. FFprobe로 메타데이터 추출 (제목, 길이)
            5. 상태를 ANALYZING 으로 변경 (다음 단계 준비 완료)
            6. 실패 시 FAILED 상태로 변경하고 에러 메시지 저장
        """
        
        project = await self.project_repo.get_by_id(project_id)
        if not project:
            return None
        
        # 상태 전환: PENDING -> DOWNLOADING
        await self.project_repo.update_status(project_id, ProjectStatus.DOWNLOADING)

        try:
            # 프로젝트 별 임시 디렉토리 생성 (temp/{project_id}/)
            output_dir = settings.temp_path / project_id
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "source.mp4"
            audio_path = output_dir / "audio.wav"

            # - yt-dlp 로 영상 다운로드 -

            # 38일차: quality 파라미터로 해상도 가변 (라벨링 480 / 발행 1080)
            # -f: 포맷 선택 (1080p 이하 MP4 영상 + M4A 오디오)
            # --merge-output-format mp4: 영상 + 오디오를 MP4로 합침
            # --no-playlist: 재생목록이면 단일 영상만 다운로드
            # --concurrent-fragments 8: 38일차 - 프래그먼트 8개 병렬 다운로드 (속도 향상)
            cmd_video = [
                "yt-dlp",
                "-f", f"bestvideo[height<={quality}][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}]",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "--concurrent-fragments", "8",
                "-o", str(video_path),
                str(project.youtube_url),
            ]

            # asyncio.create_subprocess_exec: 비동기로 외부 프로세스 실행
            # 서버의 이벤트 루프를 블로킹하지 않고 다운로드 진행
            process = await asyncio.create_subprocess_exec(
                *cmd_video,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 프로세스 완료 대기
            _, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"yt-dlp 실패: {stderr.decode()}")
            
            # - FFmpeg로 오디오 추출 -

            # -vn: 비디오 스트림 제거 (오디오만 추출)
            # -acodec pcm_s16le: 16비트 PCM 인코딩 (Whisper 입력 포맷)
            # -ar 16000: 샘플레이트 16kHz (Whisper 최적)
            # -ac 1: 모노 채널 (스테레오 불필요)
            cmd_audio = [
                "ffmpeg", "-y",         # -y: 기존 파일 덮어쓰기
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd_audio,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            # - 메타 데이터 추출 -

            meta = await self._get_video_metadata(str(video_path))

            # 상태 전황: DOWNLOADING -> ANALYZING
            # 다운로드 완료 후 분석 대기 상태로 변경
            return await self.project_repo.update(project, {
                "source_path": str(video_path),
                "audio_path": str(audio_path),
                "title": meta.get("title"),
                "duration_sec": meta.get("duration"),
                "status": ProjectStatus.ANALYZING,
            })

        except Exception as e:
            # 실패 시: 상태를 FAILED로 변경하고 에러 메시지 기록
            logger.error(f"다운로드 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(
                project_id, ProjectStatus.FAILED, str(e)
            )

            return None

    async def _get_video_metadata(self, video_path: str) -> dict:
        """
        FFprobe로 영상 메타데이터 추출

        ffprobe는 FFmpeg에 포함된 미디어 정보 분석 도구
        JSON 형식으로 출력하여 파싱

        Returns:
            {"title": "source", "duration": 1234.56}
        """
        
        cmd = [
            "ffprobe", "-v", "quiet",                            # 불필요한 출력 숨김
            "-print_format", "json",                            # JSON 형식으로 출력
            "-show_format",                                     # 포맷 정봏 (길이, 비트레이트 등) 출력
            video_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()

        try:
            info = json.loads(stdout.decode())
            fmt = info.get("format", {})

            return {
                "title": Path(video_path).stem,                 # 파일명으로 확장자 제거
                "duration": float(fmt.get("duration", 0)),      # 초 단위 길이
            }
        except (json.JSONDecodeError, ValueError):
            return {}                                           # 파실 실패 시 빈 dict 반환 (에러를 전파하지 않음)

    async def get_project(self, project_id: str) -> Optional[Project]:
        """
        프로젝트 상세 조회 (관련 쇼츠 포함).
        """

        return await self.project_repo.get_with_shorts(project_id)

    async def list_projects(self, skip: int = 0, limit: int = 20):
        """
        프로젝트 목록 조회

        Returns:
            (items, total): 프로젝트 리스트와 전체 건수 튜플
        """
        
        items = await self.project_repo.get_all(skip=skip, limit=limit)
        total = await self.project_repo.count()
        return items, total