# 계층: 비즈니스 로직 계층 (Service)
# 역할: 음성 인식 (ASR) + LLM 기반 하이라이트 추출 + 음성 없는 영상 풀백
# 의존: ProjectRepository, ShortRepository (DI로 주입 받음), gpu_manager (인프라)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, GPU 관리는 인프라 계층에 위임
# 6~7일차 : LLM 하이라이트 구현 / 11~12일차: 음성 미감지 시 시간 기반 분할 추간

"""
분석 서비스
음성 인식(ASR) 및 LLM 기반 하이라이트 추출 로직
"""

import asyncio                  # 동기 API를 스레드 풀에서 비동기로 실행
import json                     # 전사 결과 JSON 직렬화/역직렬화
from pathlib import Path        # 오디오 파일 경로 검증
from typing import Optional

from loguru import logger       # 구조화된 로깅

from app.core.config import settings

# gpu_manager에서 모델 로드/언로드 함수를 import
# 모킹 시 patch("app.services.analysis_service.load_whisper") 로 패치해야 함
from app.core.gpu_manager import load_whisper, load_llm, unload_model
from app.models.domain import Project, Shorts, ProjectStatus, ShortStatus
from app.repositories.project_repository import ProjectRepository
from app.repositories.shorts_repository import ShortsRepository

# LLM 프롬프트/호출.파싱 헬퍼 (300줄 규칙으로 분리된 모듈)
from app.services.llm_highlight_extractor import (build_highlight_prompt, call_llm, parse_highlights, create_time_based_highlights)

class AnalysisService:
    """
    분석 서비스
    책임:
        1. Whisper로 영상 음성을 텍스트로 전사 (ASR) - 구현 완료
        2. LLM으로 전사 텍스트에서 하이라이트 구간 추출 - 6~7일차 구현 예정
        3. 추출된 구간을 Shorts 엔티티로 DB에 저장

        VRAM 관리:
            - gpu_manager 모듈에 위임, 전사 완료 후 Whisper 언로드 -> LLM 로드 가능
            - 하이라이트 추출 완료 후 LLM 언로드 -> 다음 단계(YOLO) 로드 가능
    """
    
    def __init__(self, project_repo: ProjectRepository, shorts_repo: ShortsRepository):
        """DI 로 레포지토리를 주입 받음, 서비스는 레포지토리 메서드만 호출, SQL 이나 DB 세션을 직접 알지 못함"""
        
        self.project_repo = project_repo
        self.shorts_repo = shorts_repo
        self._whisper_model = None          # 지연 로딩: 필요할 때만 GPU에 로드
        self._llm_handle = None             # LLM 핸들: {"type": "openai"|"local", ...}

    # --------------------------------------------------------------
    # 공개 메서드: 음성 전사 (3~5일차)
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
        
        # 오디오 파일 존재 여부 검증
        error = self._validate_audio(project)
        if error:
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, error)
            return None

        logger.info(f"전사 시작: {project_id} | 모델: {settings.WHISPER_MODEL_SIZE} | 파일: {project.audio_path}")

        try:
            # faster-whisper는 동기 API -> 스레드 풀에서 실행하여 이벤트 루프 보호
            transcript_data = await asyncio.get_event_loop().run_in_executor(None, self._run_transcription, str(project.audio_path))

            # dict -> JSON 문자열로 변환하여 DB 저장
            transcript_json = json.dumps(transcript_data, ensure_ascii=False, indent=2)
            self._log_transcription_stats(project_id, transcript_data)

            # DB 업데이트 (상태는 ANALYZING 유지 -> 하이라이트 추출 대기)
            return await self.project_repo.update(project, {"transcript_json": transcript_json})
        except Exception as e:
            logger.error(f"전사 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, f"음성 전사 실패: {str(e)}")
            return None
        finally:
            # 성공/실패 무관하게 VRAM 해제 (다음 단계 LLM용)
            unload_model(self._whisper_model, "Whisper")
            self._whisper_model = None
    
    # --------------------------------------------------------------
    # 공개 메서드: 하이라이트 추출 (6~7일차)
    # --------------------------------------------------------------

    async def extract_highlights(self, project_id: str, max_shorts: int = 5, duration_sec: int = 60) -> list[Shorts]:
        """
        LLM 기반 하이라이트 구간 추출,

        흐름:
            1. 프롬프트 조회 + 전사 데이터 검증
            2. LLM 로드 (Gemma 4 E4B 또는 OpenAI API)
            3. 프롬프트 생성 -> LLM 호출 -> 응답 파싱
            4. Shorts 엔티티 생성 및 DB 저장
            5. 상태 전환: ANALYZING -> EDITING
            6. LLM 언로드 (VRAM 해제)
        Args:
            project_id: 분석할 프로젝트 ID
            max_shorts: 추출할 최대 쇼츠 수
            duration_sec: 각 쇼츠의 목표 길이 (초)
        Returns:
            생성된 Shorts 엔티티 목록 (실패 시 빈 리스트)
        """
        
        project = await self.project_repo.get_by_id(project_id)
        if not project:
            logger.error(f"프로젝트를 찾을 수 없음: {project_id}")
            return []
        
        # 전사 데이터(transcript_json) 존재 여부 검증
        transcript_data = self._load_transcript(project)
        if transcript_data is None:
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, "전사 데이터가 없습니다. 먼저 전사를 실행하세요.")
            return []

        logger.info(f"하이라이트 추출 시작: {project_id}")

        try:
            # 전사에 음성이 있는지 확인
            has_speech = any(seg.get("text", "").strip() for seg in transcript_data.get("segments", []))
            if has_speech:
                highlights = await asyncio.get_event_loop().run_in_executor(
                    None, self._run_highlight_extraction, transcript_data, max_shorts, duration_sec)
            else:
                # 음성 없는 영상 -> 시간 기반 균등 분할 (LLM 호출 생략)
                logger.info(f"음성 미감지 - 시간 기반 분할: {project_id}")
                total_dur = transcript_data.get("duration_sec", 0)
                highlights = create_time_based_highlights(total_dur, max_shorts, duration_sec)
            
            if not highlights:
                await self.project_repo.update_status(project_id, ProjectStatus.FAILED, "LLM이 하이라이트를 추출하지 못했습니다.")
                return []
            
            # 파싱된 하이라이트 -> Shorts 엔티티 생성 + DB 저장
            shorts_list = await self._create_shorts_entities(project_id, highlights)

            # 상태 전환: ANALYZING -> EDITING (편집 단계 준비 완료)
            await self.project_repo.update_status(project_id, ProjectStatus.EDITING)

            logger.info(f"하이라이트 완료: {project_id} | 쇼츠 {len(shorts_list)}개")
            return shorts_list
        except Exception as e:
            logger.error(f"하이라이트 추출 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, f"하이라이트 추출 실패: {str(e)}")
            return []
        finally:
            # 로컬 LLM인 경우에도 언로드 (OpenAI는 GPU 미사용)
            if self._llm_handle and self._llm_handle.get("type") == "local":
                unload_model(self._llm_handle.get("model"), "LLM")
            self._llm_handle = None
    
    async def get_shorts_by_project(self, project_id: str) -> list[Shorts]:
        """프로젝트의 쇼츠 목록 조회, 레포지토리에 위임"""
  
        return await self.shorts_repo.get_by_project(project_id)

    # --------------------------------------------------------------
    # 내부 메서드: 전사 관련
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
        beam_size=5: 빔 서치 크기 (정확도 UP, 속도 DOWN)
        word_timestamps=True: 단어 단위 타임스탬프 생성 (자막용)
        vad_filter=True: 묵음 구간 자동 제거 (처리 속도 UP)
        """

        # 모델 로드 (gpu_manager에 위임, 이미 로드 시 재사용)
        if self._whisper_model is None:
            self._whisper_model = load_whisper()

        logger.info(f"전사 실행 중: {audio_path}")
        
        segments, info = self._whisper_model.transcribe(
            audio_path, beam_size=5, word_timestamps=True, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))

        # 제너레이터 순회 -> 구조화된 dict 변환
        segments_list = []
        for segment in segments:
            segments_list.append({
                "id": segment.id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": [
                    {
                        "word": w.word.strip(), "start": round(w.start, 3),
                        "end": round(w.end, 3), "probability": round(w.probability, 4),
                    }
                    for w in (segment.words or [])
                ],
            })

        return{
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_sec": round(info.duration, 3),
            "segments": segments_list,
        }

    def _log_transcription_stats(self, project_id: str, data: dict) -> None:
        """전사 결과 통계 로깅"""

        seg_count = len(data.get("segments", []))
        word_count = sum(len(s.get("words", [])) for s in data.get("segments", []))
        logger.info(f"전사 완료: {project_id} | 세그먼트: {seg_count}개 | 단어: {word_count}개")

    # --------------------------------------------------------------
    # 내부 메서드: 하이라이트 관련 (6~7일차)
    # --------------------------------------------------------------

    def _load_transcript(self, project: Project) -> dict | None:
        """DB에서 전사 JSON 문자열을 로드하여 dict로 변환"""

        if not project.transcript_json:
            return None      
        try:
            return json.loads(project.transcript_json)
        except json.JSONDecodeError as e:
            logger.error(f"전사 JSON 파싱 실패: {e}")
            return None

    def _run_highlight_extraction(self, transcript_data: dict, max_shorts: int, duration_sec: int) -> list[dict]:
        """
        LLM 하이라이트 추출 실행 (동기 - 스레드 풀에서 호출)
        흐름: LLM 로드 -> 프롬프트 생성 -> LLM 호출 -> 응답 파싱
        각 단계는 llm_highlight_extractor 헬퍼 모듈의 함수를 사용
        """

        self._llm_handle = load_llm()       # gpu_manager에서 로드
        prompt = build_highlight_prompt(transcript_data, max_shorts, duration_sec)
        response_text = call_llm(self._llm_handle, prompt)
        total_duration = transcript_data.get("duration_sec", 0)
        return parse_highlights(response_text, total_duration, max_shorts)

    async def _create_shorts_entities(self, project_id: str, highlights: list[dict]) -> list[Shorts]:
        """
        파싱된 하이라이트 목록을 Shorts 엔티티로 변환하여 DB 저장
        각 하이라이트 dict를 Shorts SQLModel 인스턴스로 변환하고, ShortsRepository.create()를 통해 DB에 INSERT
        """

        shorts_list = []
        for h in highlights:
            shorts = Shorts(
                project_id=project_id,
                start_sec=h["start_sec"],
                end_sec=h["end_sec"],
                hook_score=h.get("hook_score"),
                highlight_reason=h.get("reason"),
                title_suggestion=h.get("title_suggestion"),
                tags_suggestion=json.dumps(h.get("tags", []), ensure_ascii=False),      # tags는 리스트 -> JSON 문자열로 변환하여 저장
                status=ShortStatus.QUEUED,                                               # 초기 상태: 편집 대기
            )
            created = await self.shorts_repo.create(shorts)                             # BaseRepository.create() 호출
            shorts_list.append(created)
            logger.info(f"쇼츠 생성: {created.id} | {h['start_sec']:.1f}-{h['end_sec']:.1f}s | 점수: {h.get('hook_score', 0):.2f}")
        
        return shorts_list