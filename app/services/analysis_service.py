# 계층: 비즈니스 로직 계층 (Service)
# 역할: 음성 인식 (ASR)과 LLM 기반 하이라이트 추출 로직
#       3~5일차: Whisper ASR 구현 완료
#       6~7일차: LLM 하이라이트 추출 구현 예정 (스텁)
# 의존: ProjectRepository, ShortRepository (DI로 주입 받음), gpu_manager (인프라)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, GPU 관리는 인프라 계층에 위임

"""
분석 서비스

음성 인식(ASR) 및 LLM 기반 하이라이브 추출 로직
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.core.gpu_manager import load_whisper, unload_model
from app.models.domain import Project, Shorts, ProjectStatus, ShortStatus
from app.repositories.project_repository import ProjectRepository
from app.repositories.shorts_repository import ShortsRepository

class AnalysisService:
    """
    분석 서비스
    
    책임:
        1. Whisper로 영상 음성을 텍스트로 전사 (ASR) - 구현 완료
        2. LLM으로 전사 텍스트에서 하이라이트 구간 추출 - 6~7일차 구현 예정
        3. 추출된 구간을 Shorts 엔티티로 DB에 저장

        VRAM 관리:
            - gpu_manager 모듈에 위임, 전사 오나료 후 Whisper 언로드 -> LLM 로드 가능
    """
    
    def __init__(
        self,
        project_repo: ProjectRepository,
        shorts_repo: ShortsRepository,
    ):
        self.project_repo = project_repo
        self.shorts_repo = shorts_repo
        self._whisper_model = None          # 지연 로딩: 필요할 때만 GPU에 로드

    # --------------------------------------------------------------
    # 공개 메서드
    # --------------------------------------------------------------

    async def transcribe(self, project_id: str) -> Optional[Project]:
        """
        Whisper를 사용한 음성 전사

        흐름: 프로젝트 조회 -> 오디오 검증 -> 전사 실행 -> JSON 저장 -> 모델 언로드

        Args:
            project_id: 전사할 프로젝트 ID

        Returns:
            업데이트 된 Project (transcript_json 포함), 실패 시 None
        """

        project = await self.project_repo.get_by_id(project_id)
        if not project:
            logger.error(f"프로젝트를 찾을 수 없음: {project_id}")
            return None
        
        # 오디오 파일 검증
        error = self._validate_audio(project)
        if error:
            await self.project_repo.update_status(
                project_id, ProjectStatus.FAILED, error
            )

            return None

        logger.info(
            f"전사 시작: {project_id} | "
            f"모델: {settings.WHISPER_MODEL_SIZE} | "
            f"파일: {project.audio_path}"
        )

        try:
            # faster-whisper는 동기 API -> 스레드 풀에서 실행하여 이벤트 루프 보호
            transcript_data = await asyncio.get_event_loop().run_in_executor(
                None, self._run_transcription, str(project.audio_path),
            )

            transcript_json = json.dumps(transcript_data, ensure_ascii=False, indent=2)
            self._log_transcription_stats(project_id, transcript_data)

            # DB 업데이트 (상태는 ANALYZING 유지 -> 하이라이트 추출 대기)
            return await self.project_repo.update(project, {"transcript_json": transcript_json, })

        except Exception as e:
            logger.error(f"전사 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(
                project_id, ProjectStatus.FAILED, f"음성 전사 실패: {str(e)}"
            )
            return None
        finally:
            # 성공/실패 무관하게 VRAM 해제 (다음 단계 LLM용)
            unload_model(self._whisper_model, "Whisper")
            self._whisper_model = None
    
    async def extract_highlights(
        self,
        project_id: str,
        max_shorts: int = 5,
        duration_sec: int = 60,       
    ) -> list[Shorts]:
        """
        LLM 기반 하이라이트 구군 추출,

        TODO (6-7일차):
            - transcript_json에서 전사 텍스트 로드
            - LLM에 프롬프트로 흥미 구간 선별 (시작/끝 시간, 선정 이유, 흥미도 점수)
            - Shorts 엔티티 생성 -> DB 저장
            - 상태 전환: ANALYZING -> EDITING
        """

        logger.info(f"하이라이트 추축 시작: {project_id}")
        raise NotImplementedError("6~7일차 구현 예정: LLM 하이라이트 분석")
    
    async def get_shorts_by_project(self, project_id: str) -> list[Shorts]:
        """
        프로젝트의 쇼츠 목록 조회, 레포지토리에 위임
        """
  
        return await self.shorts_repo.get_by_project(project_id)
    
    def _unload_model(self):
        """
        VRAM 절약을 위한 모델 언로드

        Python에서 del로 참조를 제거하면 GC가 메모리 해제
        GPU VRAM은 torch.cuda.empty_cache()도 필요할 수 있음 (구현 시 추가)
        """
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            logger.info("Whisper 모델 언로드 완료")

    # --------------------------------------------------------------
    # 내부 메서드
    # --------------------------------------------------------------

    def _validate_audio(self, project: Project) -> Optional[str]:
        """
        오디오 파일 존재 여부 검증

        Returns:
            에러 메시지 (문제 없으면 None)
        """

        if not project.audio_path:
            logger.error(f"오디오 파일 경로 없음: {project.id}")
            return "오디오 파일이 추출되지 않았습니다."
        
        if not Path(project.audio_path).exists():
            logger.error(f"오디오 파일 미존재: {project.audio_path}")
            return f"오디오 파일을 찾을 수 없습니다: {project.audio_path}"
        
        return None

    def _run_transcription(self, audio_path: str) -> dict:
        """
        faster-whisper 전사 실행 (동기 - 스레드 풀에서 호출)

        Args:
            audio_path: WAV 오디오 파일 경로

        Returns:
            {
                "language": "ko",
                "language_probability": 0.98,
                "duration_sec": 1234.56,
                "segments": [
                    {
                        "id": 0, "start": 0.0, "end": 3.5,
                        "text": "안녕하세요",
                        "words": [{"word": "안녕하세요", "start": 0.0, "end": 1.2, "probability": 0.95}]
                    }
                ]
            }
        """

        # 모델 로드 (gpu_manager에 위임, 이미 로드 시 재사용)
        if self._whisper_model is None:
            self._whisper_model = load_whisper()

        logger.info(f"전사 실행 중: {audio_path}")
        
        segments, info = self._whisper_model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # 제너레이터 순회 -> 구조화된 dict 변환
        segments_list = []
        for segment in segments:
            seg = {
                "id": segment.id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in (segment.words or [])
                ],
            }
            segments_list.append(seg)

        return{
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_sec": round(info.duration, 3),
            "segments": segments_list,
        }

    def _log_transcription_stats(self, project_id: str, data: dict) -> None:
        """
        전사 결과 통계 로깅
        """

        seg_count = len(data.get("segments", []))
        word_count = sum(len(s.get("words", [])) for s in data.get("segments", []))
        logger.info(
            f"전사 완료: {project_id} | "
            f"세그먼트: {seg_count}개 | "
            f"단어: {word_count}개 | "
            f"언어: {data.get('language', 'unknown')}"
        )