# 계층: 비즈니스 로직 계층 (Service)
# 역할: 지능형 리프레이밍, 자막 합성, GPU 가속 인코딩 로직
# 의존: ShortsRepository, ProjectRepository (DI로 주입받음)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, GPU 관리는 인프라 계층에 위임
#
# 8~10일차 변경사항:
#   - reframe_clip(): NotImplementedError 스텁 -> 실제 구현
#   - load_yolo, unload_model import 추가
#   - reframe_engine 헬퍼 모듈 import 추가
#   - _run_reframe() 내부 메서드 추가 (동기, 스레드 풀에서 실행)
#   - ProjectRepository 주입 추가 (소스 영상 경로 조회용)

"""
편집 서비스

지능형 리프레이밍, 자막 합성, GPU 가속 인코딩 로직
"""

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings

# gpu_manager에서 모델 로드/언도르 함수를 import
# 모킹 시 patch("app.services.editing_service.load_yolo") 로 패치
from app.core.gpu_manager import load_yolo, unload_model
from app.models.domain import Shorts, ShortStatus, Project
from app.repositories.shorts_repository import ShortsRepository
from app.repositories.project_repository import ProjectRepository

# 리프레이밍 헬퍼 (300줄 규칙으로 분리된 모듈)
from app.services.reframe_engine import (
    detect_subjects, smooth_trajectory, choose_strategy, build_crop_timeline, run_ffmpeg_reframe,
)

class EditingService:
    """
    편집 서비스

    책임:
        1. 16:9 -> 9:16 지능평 리프레이밍 (YOLOv8 + FFmepg CUDA)
        2. 동적 자막 생성 (ASS 포맷)
        3. 최종 인코딩(NVENC H.264)

    VRAM 전략:
        LLM 언로드 완료 후 YOLO 로드 (겹치지 않음)
        YOLOv8n (~1~2GB) + FFmpeg CUDA -> 12GB VRAM 충분
    """

    def __init__(self, shorts_repo: ShortsRepository, project_repo: ProjectRepository):
        self.shorts_repo = shorts_repo
        self.project_repo = project_repo
        self._yolo_model = None             # 지연 로딩: 필요한 때만 GPU에 로드

    async def reframe_clip(self, short_id: str, aspect_ratio: str = "9:16") -> Optional[Shorts]:
        """
        지능형 리프레이밍 (16:9 -> 9:16)

        흐름:
            1. Shorts 엔티티 조회 -> 소스 영상 경로 확인
            2. 상태 전환: QUEUED -> REFRAMING
            3. YOLO 로드 -> 피사체 탐지 -> 스무딩 -> 전략 선택 -> 크롭 타임라인
            4. FFmpeg로 리프레이밍 실행
            5. 상태 전환: REFRAMING -> QUEUED (자막/인코딩 대기)
            6. YOLO 언로드

        Args:
            short_id: 편집할 쇼츠 ID
            aspect_ratio: 목표 종횡비 (기본 "9:16")

        Returns:
            업데이트된 Shorts 엔티티, 실패 시 None
        """

        # 1. Shorts 엔티티 조회
        short = await self.shorts_repo.get_by_id(short_id)
        if not short:
            logger.error(f"쇼츠를 찾을 수 없음: {short_id}")
            return None
        
        # 소스 영상 경로 확인 (프로젝트에서 조회)
        project = await self.project_repo.get_by_id(short.project_id)
        if not project or not project.source_path:
            logger.error(f"소스 영상 없음: project={short.project_id}")
            await self._fail_short(short, "소스 영상 파일이 없습니다.")
            return None
        
        source_path = project.source_path
        if not Path(source_path).exists():
            logger.error(f"소스 파일 미존재: {source_path}")
            await self._fail_short(short, f"소스 파일을 찾을 수 없습니다: {source_path}")
            return None
        
        # 2. 상태 전환: QUEUED -> REFRAMING
        short = await self.shorts_repo.update(short, {"status": ShortStatus.REFRAMING})
        logger.info(f"리프레이밍 시작: {short_id} | {short.start_sec:.1f}-{short.end_sec:.1f}s")

        try:
            # 3~4. YOLO 탐지 + FFmpeg 리프레이밍 (동기 작업은 스레드 풀에서 실행)
            output_path = self._build_output_path(short)

            # 피사체 탐지는 동기 (YOLO predict)
            detections = await asyncio.get_event_loop().run_in_executor(
                None, self._run_detection, source_path)
            
            # 스무딩 + 전략 선택 + 크롭 타임라인 (동기, 경량)
            detections = smooth_trajectory(detections)
            strategy = choose_strategy(detections)
            timeline = build_crop_timeline(detections, strategy, aspect_ratio)

            logger.info(f"리프레이밍 전략: {strategy} | 쇼츠: {short_id}")

            # FFmpeg 실행 (비동기 서브프로세스)
            success = await run_ffmpeg_reframe(source_path, str(output_path), timeline)

            if not success:
                await self._fail_short(short, "FFmpeg 리프레이밍 실패")
                return None
            
            # 5. 상태 전환 + 출력 경로 저장
            # 자막/인코딩은 11~12일차에 구현, 현재는 QUEUED로 유지
            short = await self.shorts_repo.update(short, {
                "output_path": str(output_path),
                "status": ShortStatus.QUEUED,
            })

            logger.info(f"리프레이밍 완료: {short_id} -> {output_path}")
            return short
        
        except Exception as e:
            logger.error(f"리프레이밍 실패 [{short_id}]: {e}")
            await self._fail_short(short, f"리프레이밍 실패: {str(e)}")
            return None
        finally:
            # 6. YOLO 언로드 (성공/실패 무관)
            unload_model(self._yolo_model, "YOLO")
            self._yolo_model = None

    # --------------------------------------------------------------
    # 내부 메서드
    # --------------------------------------------------------------

    def _run_detection(self, source_path: str) -> list[dict]:
        """
        YOLO 피사체 탐지 실행 (동기 - 스레드 풀에서 호출)
        """
        
        if self._yolo_model is None:
            self._yolo_model = load_yolo()
        return detect_subjects(self._yolo_model, source_path)

    def _build_output_path(self, short: Shorts) -> Path:
        """
        리프레이밍 결과 파일 경로 생성
        """

        output_dir = settings.temp_path / short.project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"reframed_{short.id}.mp4"
    
    async def _fail_short(self, short: Shorts, error_msg: str) -> None:
        """
        쇼츠 상태를 FAILED로 변경
        """

        logger.error(f"쇼츠 실패: {short.id} | {error_msg}")
        await self.shorts_repo.update(short, {"status": ShortStatus.FAILED})

    # --------------------------------------------------------------
    # 스텁 메서드 (11~12일차 구현 예정)
    # --------------------------------------------------------------

    async def generate_subtitles(self, shorts_id: str) -> Optional[Shorts]:
        """
        동적 자막 생성 (ASS 포맷)

        TODO (2주차 11~12일차)
            - Whisper의 단어 타임스탬프 기반으로 자막 생성
            - 현재 발화 중인 단어를 강조 (색상 변경, 크기 확대)
            - LLM 감정 분석 -> 강조할 키워드 자동 선택
            - ASS(Advanced Substation Alpha) 포맥으로 렌더링 (위치, 폰트, 크기, 색상, 애니메이션 효과 지원)

        ASS 포맷을 사용하는 이유:
            - SRT 보다 훨씬 풍부한 스타일링 가능
            - 위치 지정, 색상 변화, 페이드인/아웃 등 애니메이션 지원
            - FFmpeg의 subtitles 필터로 직접 합성 가능
        """

        logger.info(f"자막 생성 시작: {shorts_id}")
        raise NotImplementedError("2주차 구현 예정: 동적 자막")

    async def encode_final(self, shorts_id: str) -> Optional[Shorts]:
        """
        최종 인코딩 (NVENC H.264)

        TODO (2주차 11~12일차)
            - 리프레이밍된 영상 + 자막 + 오디오를 합성
            - NVIDIA NVENC 하드웨어 인코더로 H.264 인코딩 (CPU 인코딩 대비 3~5배 빠름)
            - 오디오 노멀라이즈 (-14 LUFS, 유부트 표준)
            - 피치 보존 속도 조장 (1.05x, 자연스러운 템포 가속)
            - 사일런스 리무버 (묵음 구간 자동 제거)
            - 최종 출력: 9:16 종횡비, 1080x1920, H.264, AAC

        FFmpeg 명령어 구성 예시:
        ffmpeg  -hwaccel cuda -i source.mp4
                -vf "crop_cuda-...,scale_cuda=1089:1920"
                -c:v h264_nvenc -preset p4 -cq 23
                -c:a aac -b:a 128k
                output.mp4
        """

        logger.info(f"최종 인코딩 시작: {shorts_id}")
        raise NotImplementedError("2주차 구현 예정: NVENC 인코딩")