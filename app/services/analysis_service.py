# 계층: 비즈니스 로직 계층 (Service)
# 역할: 음성 인식 (ASR) + LLM 기반 하이라이트 추출 + 음성 없는 영상 폴백
# 의존: ProjectRepository, ShortRepository (DI로 주입 받음), gpu_manager (인프라)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, GPU 관리는 인프라 계층에 위임
# 6~7일차 : LLM 하이라이트 구현 / 11~12일차: 음성 미감지 시 시간 기반 분할 추가
# 13일차: duration_sec 파라미터 제거 - LLM 자동 길이 판단으로 전환
# 14~15일차: VLM 멀티모달 분기 추가 - 영상 프레임 + 텍스트 통합 분석
# 17일차: 
#   - LLM 로드 / 언로드를 extract_highlight() 에만 둠 (청크 루프 중 실수 방지)
#   - 청크 루프 + try/except 격리 (실패율 절반 초과 시 전체 실패)
#   - transcript_chunker 사용 (split_transcript_into_chunks + merge_and_rerank_highlights)
# 22일차: LoRA 어댑터 로드 분기 추가 (LORA_ENABLED=True 시 파인튜닝된 모델 사용)

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

# gpu_manager: 모킹 시 patch("app.services.analysis_service.load_whisper") 로 패치
from app.core.gpu_manager import load_whisper, load_llm, unload_model
from app.models.domain import Project, Shorts, ProjectStatus, ShortStatus
from app.repositories.project_repository import ProjectRepository
from app.repositories.shorts_repository import ShortsRepository

# LLM 프롬프트/호출.파싱 헬퍼 (300줄 규칙으로 분리된 모듈)
from app.services.llm_highlight_extractor import (build_highlight_prompt, call_llm, parse_highlights, create_time_based_highlights)

# VLM 멀티모달 분석 (14~15일차): 영상 프레임 + 텍스트 통합 분석
from app.services.vlm_client import is_vlm_available, run_vlm_analysis
# 청크 분할 _ 재랭킹 헬퍼 (17일차 신규)
from app.services.transcript_chunker import(split_transcript_into_chunks, merge_and_rerank_highlights)

# 17일차: 한청크당 후보 수 배수 / 청크당 평균 LLM 처리 시간 추정 (로그용, 초)
_CANDIDATE_MULTIPLIER = 2
_EST_SEC_PER_CHUNK = 20

def _serialize_segment(segment) -> dict:
    """faster-whisper Segment 객체를 dict로 직렬화 (words 포함)"""

    return {
        "id": segment.id,
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "text": segment.text.strip(),
        "words": [
            {"word": w.word.strip(), "start": round(w.start, 3),
             "end": round(w.end, 3), "probability": round(w.probability, 4)}
            for w in (segment.words or [])
        ],
    }

class AnalysisService:
    """분석 서비스 - Whisper 전사 + LLM/VLM 하이라이트 추출 (순차 모델 스위칭)"""
    
    def __init__(self, project_repo: ProjectRepository, shorts_repo: ShortsRepository):
        """DI 로 레포지토리를 주입"""
        
        self.project_repo = project_repo
        self.shorts_repo = shorts_repo
        self._whisper_model = None          # 지연 로딩: 필요할 때만 GPU에 로드
        self._llm_handle = None             # LLM 핸들: {"type": "openai"|"local", ...}

    # 공개 메서드: 음성 전사 (3~5일차)
    async def transcribe(self, project_id: str) -> Optional[Project]:
        """Whisper를 사용한 음성 전사 (흐름: 검증 -> 전사 -> JSON 저장 -> 모델 언로드)"""

        project = await self.project_repo.get_by_id(project_id)
        if not project:
            logger.error(f"프로젝트를 찾을 수 없음: {project_id}")
            return None
        
        error = self._validate_audio(project)
        if error:
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, error)
            return None

        logger.info(f"전사 시작: {project_id} | 모델: {settings.WHISPER_MODEL_SIZE} | 파일: {project.audio_path}")

        try:
            transcript_data = await asyncio.get_event_loop().run_in_executor(None, self._run_transcription, str(project.audio_path))
            transcript_json = json.dumps(transcript_data, ensure_ascii=False, indent=2)
            self._log_transcription_stats(project_id, transcript_data)

            return await self.project_repo.update(project, {"transcript_json": transcript_json})
        except Exception as e:
            logger.error(f"전사 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, f"음성 전사 실패: {str(e)}")
            return None
        finally:
            unload_model(self._whisper_model, "Whisper")
            self._whisper_model = None
    
    # 공개 메서드: 하이라이트 추출 (6~7/14~15/17일차)
    async def extract_highlights(self, project_id: str, max_shorts: int = 5) -> list[Shorts]:
        """
        하이라이트 추출: VLM 우선 -> 텍스트 LLM 폴백(17일차: 청크 분할) -> 음성 없으면 시간 분할
        13일차: duration_sec 제거(LLM 자동 판단) / 14~15일차: VLM 분기 / 17일차: 청크 분할
        """
        
        project = await self.project_repo.get_by_id(project_id)
        if not project:
            logger.error(f"프로젝트를 찾을 수 없음: {project_id}")
            return []
        
        transcript_data = self._load_transcript(project)
        if transcript_data is None:
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, "전사 데이터가 없습니다. 먼저 전사를 실행하세요.")
            return []

        logger.info(f"하이라이트 추출 시작: {project_id}")

        try:
            has_speech = any(seg.get("text", "").strip() for seg in transcript_data.get("segments", []))

            # VLM 우선(14~15일차) -> LoRA VLM(22일차) -> 텍스트 LLM 청크 분할 (17일차) -> 시간 분할
            # LORA_ENABLED=true 시 vlm_client 내부에서 생성기 -> 판별기 파이프라인 자동 실행
            if is_vlm_available() and project.source_path:
                highlights = await run_vlm_analysis(project.source_path, transcript_data, max_shorts)
            elif has_speech:
                # 17일차: LLM 로드 책임을 여기로 이동 (청크 루프 중 실수 방지)
                self._llm_handle = load_llm()
                highlights = await asyncio.get_event_loop().run_in_executor(
                    None, self._run_highlight_extraction, transcript_data, max_shorts)
            else:
                logger.info(f"음성 미감지 - 시간 기반 분할: {project_id}")
                total_dur = transcript_data.get("duration_sec", 0)
                highlights = create_time_based_highlights(total_dur, max_shorts)
            
            if not highlights:
                await self.project_repo.update_status(project_id, ProjectStatus.FAILED, "LLM이 하이라이트를 추출하지 못했습니다.")
                return []
            
            shorts_list = await self._create_shorts_entities(project_id, highlights)
            await self.project_repo.update_status(project_id, ProjectStatus.EDITING)
            logger.info(f"하이라이트 완료: {project_id} | 쇼츠 {len(shorts_list)}개")

            return shorts_list
        except Exception as e:
            logger.error(f"하이라이트 추출 실패 [{project_id}]: {e}")
            await self.project_repo.update_status(project_id, ProjectStatus.FAILED, f"하이라이트 추출 실패: {str(e)}")
            return []
        finally:
            # 17일차: LLM 언로드 1회만 (OpenAI는 GPU 미사용)
            if self._llm_handle and self._llm_handle.get("type") == "local":
                unload_model(self._llm_handle.get("model"), "LLM")
            self._llm_handle = None
    
    async def get_shorts_by_project(self, project_id: str) -> list[Shorts]:
        """프로젝트의 쇼츠 목록 조회, 레포지토리에 위임"""
  
        return await self.shorts_repo.get_by_project(project_id)

    # 내부 메서드: 전사 관련
    def _validate_audio(self, project: Project) -> Optional[str]:
        """오디오 파일 존재 여부 검증, 에러 메시지 (문제 없으면 None)"""

        if not project.audio_path:
            logger.error(f"오디오 파일 경로 없음: {project.id}")
            return "오디오 파일이 추출되지 않았습니다."
        
        if not Path(project.audio_path).exists():
            logger.error(f"오디오 파일 미존재: {project.audio_path}")
            return f"오디오 파일을 찾을 수 없습니다: {project.audio_path}"
        
        return None

    def _run_transcription(self, audio_path: str) -> dict:
        """faster-whisper 전사 실행 (beam_size=5, word_timestamps=True, vad_filter=True)"""

        if self._whisper_model is None:
            self._whisper_model = load_whisper()

        logger.info(f"전사 실행 중: {audio_path}")
        segments, info = self._whisper_model.transcribe(
            audio_path, beam_size=5, word_timestamps=True, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))

        return{
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_sec": round(info.duration, 3),
            "segments": [_serialize_segment(s) for s in segments],
        }

    def _log_transcription_stats(self, project_id: str, data: dict) -> None:
        """전사 결과 통계 로깅"""

        seg_count = len(data.get("segments", []))
        word_count = sum(len(s.get("words", [])) for s in data.get("segments", []))
        logger.info(f"전사 완료: {project_id} | 세그먼트: {seg_count}개 | 단어: {word_count}개")


    # 내부 메서드: 하이라이트 관련 (6~7/17일차)
    def _load_transcript(self, project: Project) -> dict | None:
        """DB에서 전사 JSON 문자열을 로드하여 dict로 변환"""

        if not project.transcript_json:
            return None      
        try:
            return json.loads(project.transcript_json)
        except json.JSONDecodeError as e:
            logger.error(f"전사 JSON 파싱 실패: {e}")
            return None

    def _run_highlight_extraction(self, transcript_data: dict, max_shorts: int) -> list[dict]:
        """청크 분할 기반 하이라이트 추출 (17일차, 전제: self._llm_handle 로드 상태)"""

        if self._llm_handle is None:
            raise RuntimeError("LLM 미로드 extract_highlights() 통해 호출하세요.")
        
        chunks = split_transcript_into_chunks(
            transcript_data,
            chunk_duration_sec=settings.CHUNK_DURATION_SEC,
            overlap_sec=settings.CHUNK_OVERLAP_SEC
        )
        logger.info(
            f"청크 {len(chunks)}개 처리 시작 | 예상 GPU 점유: 약 "
            f"{len(chunks) * _EST_SEC_PER_CHUNK}초"
        )

        per_chunk_max = max_shorts * _CANDIDATE_MULTIPLIER
        chunk_results, failed_chunk = self._process_chunks(chunks, per_chunk_max)

        if len(failed_chunk) > len(chunks) / 2:
            raise RuntimeError(f"청크 절반 이상 실패 ({len(failed_chunk)}/{len(chunks)})")
        
        return merge_and_rerank_highlights(chunk_results, max_shorts, iou_threshold=settings.HIGHLIGHT_IOU_THRESHOLD)

    def _process_chunks(self, chunks: list[dict], per_chunk_max: int) -> tuple[list[list[dict]], list[int]]:
        """청크별 LLM 호출 + 예외 격리 (17일차 신규), Returns: (chunk_results, failed_indices)"""
        
        chunk_results: list[list[dict]] = []
        failed_chunks: list[int] = []
        total = len(chunks)
        for chunk in chunks:
            idx = chunk["chunk_index"]
            try:
                logger.info(f"청크 {idx + 1}/{total} 처리 중")
                prompt = build_highlight_prompt(chunk, per_chunk_max)
                response_text = call_llm(self._llm_handle, prompt)
                highlights = parse_highlights(
                    response_text, chunk["end_offset_sec"], per_chunk_max,
                    chunk_start=chunk["start_offset_sec"],
                )
                chunk_results.append(highlights)
                self._log_chunk_stats(idx, total, highlights)
            except Exception as e:
                logger.warning(f"청크 {idx + 1}/{total} 처리 실패: {e}")
                failed_chunks.append(idx)
                chunk_results.append([])

        return chunk_results, failed_chunks

    def _log_chunk_stats(self, idx: int, total: int, highlights: list[dict]) -> None:
        """청크별 점수 분포 로그 (17일차, 관측용)"""
        
        if not highlights:
            logger.info(f"청크 {idx + 1}/{total} 결과: 0개 후보")
            return
        scores = [h.get("hook_score", 0) for h in highlights]
        logger.info(
            f"청크 {idx + 1}/{total} 결과: {len(highlights)}개 후보 | "
            f"평균: {sum(scores)/len(scores):.2f} | 최고: {max(scores):.2f}"
        )

    async def _create_shorts_entities(self, project_id: str, highlights: list[dict]) -> list[Shorts]:
        """파싱된 하이라이트 목록을 Shorts 엔티티로 변환하여 DB 저장"""

        shorts_list = []
        for h in highlights:
            shorts = Shorts(
                project_id=project_id,
                start_sec=h["start_sec"], end_sec=h["end_sec"],
                hook_score=h.get("hook_score"),
                highlight_reason=h.get("reason"),
                title_suggestion=h.get("title_suggestion"),
                tags_suggestion=json.dumps(h.get("tags", []), ensure_ascii=False),          # tags는 리스트 -> JSON 문자열로 변환하여 저장
                aspect_ratio=h.get("aspect_ratio", "9:16"),                                 # 21일차: AI 추천 종횡비
                status=ShortStatus.QUEUED,                                                  # 초기 상태: 편집 대기
            )
            created = await self.shorts_repo.create(shorts)                             
            shorts_list.append(created)
            logger.info(f"쇼츠 생성: {created.id} | {h['start_sec']:.1f}-{h['end_sec']:.1f}s | 점수: {h.get('hook_score', 0):.2f}")
        
        return shorts_list