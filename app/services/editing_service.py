# 계층: 비즈니스 로직 계층 (Service)
# 역할: 지능형 리프레이밍, 자막 합성, GPU 가속 인코딩 로직
# 의존: ShortsRepository, ProjectRepository (DI로 주입받음)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, GPU 관리는 인프라 계층에 위임
#
# 8~10일차: reframe_clip() 구현
# 11~12일차 변경사항:
#   - reframe_clip: extract_clip() 클립 추출 단계 추가 (치명적 버그 수정)
#   - 전체 소스가 아닌 start_sec - end_sec 클립에서만 YOLO/FFmpeg 실행
#   - generate_subtitle(): 스텁 -> ASS 동적 자막 생성 구현
#   - encode_final(): 스텁 -> NVENC 최종 인코딩 구현
#   - subtitle_generator 헬퍼 모듈 import 추가
# 17일차:
#   - generate_subtitle() 시작부에 verify_font() 호출 추가
#   - 퐅느 미설치 시 명확한 에러 메시지로 FAILED 처리 (한글 tofu 방지)

"""
편집 서비스
지능형 리프레이밍, 자막 합성, GPU 가속 인코딩 로직
"""

import asyncio
import json
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

# 리프레이밍 헬퍼 (estract_clip 추가됨, 11~12일차)
from app.services.reframe_engine import (
    extract_clip, detect_subjects, smooth_trajectory, 
    choose_strategy, build_crop_timeline, run_ffmpeg_reframe,
)

# 자막/인코딩 헬퍼 (11~12일차 신규 / 17일차: verify_font 추가)
from app.services.subtitle_generator import (
    extract_words_for_range, build_ass_header, build_ass_events,
    write_ass_file, run_ffmpeg_subtitle, run_ffmpeg_encode, verify_font
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

    # --------------------------------------------------------------
    # 리프레이밍 (8~9일차 / 11~12일차 클립 추출 추가)
    # --------------------------------------------------------------
    async def reframe_clip(self, short_id: str, aspect_ratio: str = "9:16") -> Optional[Shorts]:
        """
        클립 추출 + 지능형 리프레이밍 (16:9 -> 9:16)

        수정 사항 (11~12일차):
            기존: 전체 소스(30분)에 YOLO+FFmpeg 실행 -> 시간 낭비 + 결과물 오류
        수정: extract_clip()으로 start_sec ~ end_sec 만 추출 후 처리

        흐름:
            1. Shorts/Project 조회 -> 소스 영상 경로 확인
            2. FFmpeg로 클립 추출 (start_sec ~ end_sec)
            3. YOLO 로 클립 내 피사체 탐지 
            4. 스무딩 -> 전략 선택 -> 크롭 타임라인
            5. FFmpeg로 크롭 리프레이밍
            6. YOLO 언로드
        Args:
            short_id: 편집할 쇼츠 ID
            aspect_ratio: 목표 종횡비 (기본 "9:16")
        Returns:
            업데이트된 Shorts 엔티티, 실패 시 None
        """

        # Shorts 엔티티 조회
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
        
        if not Path(project.source_path).exists():
            logger.error(f"소스 파일 미존재: {project.source_path}")
            await self._fail_short(short, f"소스 파일을 찾을 수 없습니다: {project.source_path}")
            return None
        
        # 상태 전환: QUEUED -> REFRAMING
        short = await self.shorts_repo.update(short, {"status": ShortStatus.REFRAMING})
        logger.info(f"리프레이밍 시작: {short_id} | {short.start_sec:.1f}-{short.end_sec:.1f}s")

        try:
            # [핵심 수정] 1단계: 클립 추출 - 전체 소스에서 구간만 잘라냄
            clip_path = self._build_output_path(short, "clip")
            clip_ok = await extract_clip(project.source_path, str(clip_path), short.start_sec, short.end_sec)
            if not clip_ok:
                await self._fail_short(short, "클립 추출 실패")
                return None

            # 2단계 클립에서 YOLO 피사체 탐지 (전체 소스 X, 클립 O)
            output_path = self._build_output_path(short, "reframed")
            detections = await asyncio.get_event_loop().run_in_executor(
                None, self._run_detection, str(clip_path))
            
            # 3단계: 스무딩 -> 전략 + 크롭 타임라인
            detections = smooth_trajectory(detections)
            strategy = choose_strategy(detections)
            timeline = build_crop_timeline(detections, strategy, aspect_ratio)
            logger.info(f"리프레이밍 전략: {strategy} | 쇼츠: {short_id}")

            # 4단계: FFmpeg 리프레이밍 (클립 입력)
            success = await run_ffmpeg_reframe(str(clip_path), str(output_path), timeline)
            if not success:
                await self._fail_short(short, "FFmpeg 리프레이밍 실패")
                return None
            
            short = await self.shorts_repo.update(short, {
                "output_path": str(output_path), "status": ShortStatus.QUEUED})

            logger.info(f"리프레이밍 완료: {short_id} -> {output_path}")
            return short
        
        except Exception as e:
            logger.error(f"리프레이밍 실패 [{short_id}]: {e}")
            await self._fail_short(short, f"리프레이밍 실패: {str(e)}")
            return None
        finally:
            unload_model(self._yolo_model, "YOLO")
            self._yolo_model = None

    # --------------------------------------------------------------
    # 자막 생성 (11~12일차)
    # --------------------------------------------------------------
    async def generate_subtitles(self, short_id: str) -> Optional[Shorts]:
        """
        ASS 동적 자막 생성 + FFmpeg 합성
        단어 타임스탬프는 클립 기준 0초부터 시작 (extract_words_for_range가 오프셋 처리)

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

        short = await self.shorts_repo.get_by_id(short_id)
        if not short:
            return None
        
        if not short.output_path or not Path(short.output_path).exists():
            await self._fail_short(short, "리프레이밍된 영상이 없습니다. 먼저 edit을 실행하세요.")
            return None
        
        project = await self.project_repo.get_by_id(short.project_id)
        transcript = self._load_transcript(project)
        if transcript is None:
            await self._fail_short(short, "전사 데이터가 없습니다.")
            return None
        
        logger.info(f"자막 생성 시작: {short_id} | {short.start_sec:.1f}-{short.end_sec:.1f}s")

        try:
            words = extract_words_for_range(transcript, short.start_sec, short.end_sec)

            # 음성 없는 영상 - 단어가 없으면 자막 단계 건너뛰기
            if not words:
                logger.info(f"자막 생략 (음성 없음): {short_id}")
                return short
            
            # 17일차: 자막 렌더링 전 폰트 설치 확인 (한글 tofu 방지)
            try:
                if not verify_font():
                    await self._fail_short(short, "한글 폰트(Noto Sans CJK KR) 미설치. 'sudo apt install -y fonts-noto-cjk fonts-noto-cjk-extra' 실행 후 재시도")
                    return None
            except RuntimeError as e:
                await self._fail_short(short, f"폰트 검증 실패 {e}")
                return None

            header = build_ass_header()
            events = build_ass_events(words)

            ass_path = self._build_output_path(short, "subtitle", ext=".ass")
            if not write_ass_file(str(ass_path), header, events):
                await self._fail_short(short, "ASS 자막 파일 생성 실패")
                return None
            
            subtitled_path = self._build_output_path(short, "subtitled")
            success = await run_ffmpeg_subtitle(short.output_path, str(ass_path), str(subtitled_path))
            if not success:
                await self._fail_short(short, "FFmpeg 자막 합성 실패")
                return None
            
            short = await self.shorts_repo.update(short, {"output_path": str(subtitled_path)})
            logger.info(f"자막 생성 완료: {short_id} -> {subtitled_path}")
            return short
        except Exception as e:
            logger.error(f"자막 생성 실패 [{short_id}]: {e}")
            await self._fail_short(short, f"자막 생성 실패: {str(e)}")
            return None

    # --------------------------------------------------------------
    # 최종 인코딩 (11~12일차)
    # --------------------------------------------------------------
    async def encode_final(self, shorts_id: str) -> Optional[Shorts]:
        """NVENC H.264 인코딩 + loudnorm(`14 LUFS) + atempo(1.05x) -> COMPLETED"""

        short = await self.shorts_repo.get_by_id(shorts_id)
        if not short:
            return None
        
        if not short.output_path or not Path(short.output_path).exists():
            await self._fail_short(short, "인코딩할 영상이 없습니다.")
            return None
        
        short = await self.shorts_repo.update(short, {"status": ShortStatus.ENCODING})
        logger.info(f"최종 인코딩 시작: {shorts_id}")

        try:
            final_path = settings.output_path / f"{shorts_id}.mp4"
            success = await run_ffmpeg_encode(short.output_path, str(final_path))
            if not success:
                await self._fail_short(short, "FFmpeg 최종 인코딩 실패")
                return None
            
            short = await self.shorts_repo.update(short, {
                "output_path": str(final_path), "status": ShortStatus.COMPLETED,
            })

            logger.info(f"최종 인코딩 완료: {shorts_id} -> {final_path}")
            return short
        except Exception as e:
            logger.error(f"최종 인코딩 실패 [{shorts_id}]: {e}")
            await self._fail_short(short, f"최종 인코딩 실패: {str(e)}")
            return None

    # --------------------------------------------------------------
    # 내부 메서드
    # --------------------------------------------------------------

    def _run_detection(self, source_path: str) -> list[dict]:
        """YOLO 피사체 탐지 실행 (동기 - 스레드 풀에서 호출)"""
        
        if self._yolo_model is None:
            self._yolo_model = load_yolo()
        return detect_subjects(self._yolo_model, source_path)

    def _build_output_path(self, short: Shorts, prefix: str, ext: str = ".mp4") -> Path:
        """중간/최종 결과 파일 경로 생성"""

        output_dir = settings.temp_path / short.project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{prefix}_{short.id}{ext}"
    
    def _load_transcript(self, project: Optional[Project]) -> Optional[dict]:
        """프로젝트의 전사 JSON을 dict로 변환"""

        if not project or not project.transcript_json:
            return None
        
        try:
            return json.loads(project.transcript_json)
        except json.JSONDecodeError as e:
            logger.error(f"전사 JSON 파싱 실패: {e}")
            return None
    
    async def _fail_short(self, short: Shorts, error_msg: str) -> None:
        """쇼츠 상태를 FAILED로 변경"""

        logger.error(f"쇼츠 실패: {short.id} | {error_msg}")
        await self.shorts_repo.update(short, {"status": ShortStatus.FAILED})