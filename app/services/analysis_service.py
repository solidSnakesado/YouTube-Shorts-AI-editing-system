# 계층: 비즈니스 로직 계층 (Service)
# 역할: 음성 인식 (ASR)과 LLM 기반 하이라이트 추출 로직
#       현재는 스텁(stub) 상태 -> 3~7일 차에 실제 구현 예정
# 의존: ProjectRepository, ShortRepository (DI로 주입 받음)
# MVA 원칙: 의도적 코드 수준 부채 - 구조만 잡고 구현은 나중에
#
# 3~5일차 구현 항목: Whsper ASR (faster-whisper)
# 6~7일차 구현 항목: LLM 하이라이트 추출

"""
분석 서비스

음성 인식(ASR) 및 LLM 기반 하이라이브 추출 로직
1주차 개발의 핵심 구현 대상
"""

from typing import Optional

from loguru import logger

from app.models.domain import Project, Shorts, ProjectStatus, ShortStatus
from app.repositories.project_repository import ProjectRepository
from app.repositories.shorts_repository import ShortsRepository

class AnalysisService:
    """
    분석 서비스
    
    책임:
        1. Whisper로 영상 음성을 텍스트로 전사 (ASR)
        2. LLM으로 전사 텍스트에서 하이라이트 구간 추출
        3. 추출된 구간을 Shorts 엔티티로 DB에 저장

        VRAM 관리:
            - Whisper 모델과 LLM을 동시에 GPU에 올릴 수 없으므로 (VRAM 제한)
            - 모델 스위칭 전략 사용: 전사 끝 -> Whisper 언로드 -> LLM 로드
    """
    
    def __init__(
        self,
        project_repo: ProjectRepository,
        shorts_repo: ShortsRepository,
    ):
        self.project_repo = project_repo
        self.shorts_repo = shorts_repo
        self._whisper_model = None          # 지연 로딩: 필요할 때만 GPU에 로드

    async def transcribe(self, project_id: str) -> Optional[Project]:
        """
        Whisper를 사용한 음성 전사

        TODO (3~5일차):
            - faster-whisper 모델 로드 (setting.WHISPER_MODEL_SIZE)
            - project.audio_path의 WAV 파일을 입력으로 전사
            - 단어 단위 타임스탬프 추출 (word_timestamps=True)
            - 결과를 JSON으로 변환하여 project.transcript_json에 저장
            - 상태 전환: ANALYZING 유지 (하이라이트 추출 대기)

        구현 시 참고:
            - faster-whisper는 CTranslate2 기반으로 표준 Whisper 대비 수십 배 빠름
            - VRAM 사용량: medium 모델 ~5GB, large-v3-turbo ~8GB
            - RTX 5070 Ti (11.9GB VRAM)에서 large-v3-turbe 사용 가능
        """

        logger.info(f"전사 시작: {project_id}")
        raise NotImplementedError("3~5일차 구현 예정: Whisper ASR")
    
    async def extract_highlights(
        self,
        project_id: str,
        max_shorts: int = 5,
        duration_sec: int = 60,       
    ) -> list[Shorts]:
        """
        LLM 기반 하이라이트 구군 추출,

        TODO (6-7일차):
            - project.transcript_json에서 전사 텍스트 로드
            - LLM에 프롬프트 전달:
                "이 영상에서 가장 흥미로운 {max_shorts}개 구간을 선별하고,
                각 구간의 시작/끝 시간(초)과 선정 이유를 JSON으로 반환하라"
            - LLM 응답 파싱 -> 시작/끝 타임스탬프, 선정 이유, 흥미도 점수 추출
            - 각 구간을 Shorts 엔티티로 생성하여 DB에 저장
            - 상태 전환: ANALYZING -> EDITING (편집 대기)

        구현 시 참고:
            - 로컬 LLM: llama-cpp-python (GGUF 포맷)
            - 클라우드 LLM: OpenAI GPT-4o-mini (OPENAI_API_KEY가 있을 때), 또는 다른 API도 사용 가능
            - Whisper 전사 후 모델 언로드 -> LLM 로드 (VRAM 스위칭)
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