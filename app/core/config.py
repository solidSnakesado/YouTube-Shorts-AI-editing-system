# 계층: 인프라 계층 (Core)
# 역할: 애플리케이션 전역 환경 설정을 중앙 관리 (.env -> Python 타입 자동 변환)
# 의존: 없음 (최하위 인프라 모듈)
# 변경 이력:
#   6~7일차:    LLM 설정 (GGUF 파일명, GPU 오프로드, 컨텍스트 크기)
#   14일차:     VLM 서버 + 프레임 추출 설정
#   17일차:     청크 분할 설정 (장편 영상 대응)
#   20일차:     히트맵 수집 설정 (HEATMAP_*)
#   21일차:     파인튜닝 데이터 준비 설정 (FINETUNE_*)
#   22일차:     QLoRA 파인튜닝 설정 (LORA_*)
#               Qwen2.5-VL-7B 모델 전환 (학습 + 추론 통일, Gemma 4 E4B VRAM 초과 대응)

"""
환경 설정 모듈

모든 환경 변수를 Pydantic Setting로 관리한다.
.env 파일에서 값을 읽어오며, 타입 검증이 자동으로 수행된다.
"""

from pathlib import Path            # 파일 경로를 OS 독립적으로 처리하기 위한 표준 라이브러리
from functools import lru_cache     # 함수 호출 결과를 캐싱하여 동일 설정 객체를 재사용 (싱글턴 패턴)
from typing import Literal          # 허용되는 값의 목록을 타입 수준에서 제한

# pydantic_settings:    .env 파일을 자동으로 읽어서 클래스 필드에 매핑해주는 라이브러리
# BaseSettings:         .env와 환경 변수를 자동으로 바인딩하는 기반 클래스
# SettingsConfigDict:   .env 파일 경로, 인코딩 등 설정 동작을 커스터마이징
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    애플리케이션 전역 설정 클래스

    이 클래스의 각 필드는 .env 파일의 키와 1:1 매핑
        ex> APP_NAME="YT Shorts AI" -> self.APP_NAME = "YT Shorts AI"

    타입 힌트(str, int, bool 등)에 따라 자동 변환되므로,
    .env에 문자열로 "8000"이라고 작성되어 있어도 APP_PORT는 int(8000)이 됨
    """

    # model_config: Pydantic Settings의 동작 방식을 정의
    model_config = SettingsConfigDict(
        env_file=".env",                # 프로젝트 루트의 .env 파일에서 값을 읽음
        env_file_encoding="utf-8",      # 한글이 포함된 값도 정상 처리
        case_sensitive=True,            # 환경 변수명의 대소문자를 구분 (APP_NAME과 app_name 을 구분)
    )

    # --------------------------------------------------------------
    # 애플리케이션 기본 설정
    # --------------------------------------------------------------
    APP_NAME: str = "YT Shorts AI"                                              # Swagger UI 제목에 표시
    APP_ENV: Literal["development", "staging", "production"] = "development"    # Literal 타입: 이 세 값 중 하나만 허용, 그 외 값이 들어오면 서버 시작시 ValidationError 발생
    APP_DEBUG: bool = True                                                      # True 이면 SQLAlchemy 쿼리 로그 출력 등 디버그 모드 활성화
    APP_HOST: str = "0.0.0.0"                                                   # 0.0.0.0 = 모든 네트워크 인터페이스에서 접속 허용
    APP_PORT: int = 8000                                                        # Uvicorn 서버 포트

    # --------------------------------------------------------------
    # 데이터 베이스
    # --------------------------------------------------------------
    # sqlite+aiosqlite: 비동기 SQLite 드라이버
    #  ///./data/shorts_ai.db: 프로젝트 루트 기준 상대 경로 (슬래시 3개 -> 상대경로)
    # 프로덕션 전환 시: postgresql+asyncpg://user:pass@host:5432/dbname 으로 교체
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/shorts_ai.db"

    # --------------------------------------------------------------
    # 보안
    # --------------------------------------------------------------
    SECRET_KEY: str = "키 입력 필요"                 # JWT 서명 키
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60          # 토큰 만료 시간 (분)
    ALGORITHM: str = "HS256"                       # JWT 서명 알고리즘 (HMAC-SHA256)

    # --------------------------------------------------------------
    # AI 모델 - Whisper (faster-whisper)
    # --------------------------------------------------------------
    WHISPER_MODEL_SIZE: str = "medium"              # tiny|base|small|medium|large-v3-turbo
    WHISPER_DEVICE: str = "cuda"                    # cuda(GPU) 또는 cpu
    WHISPER_COMPUTE_TYPE: str = "float16"           # float16(빠름) 또는 int8(VRAM 절약)

    # --------------------------------------------------------------
    # AI 모델 - YOLO (ultralytics)
    # --------------------------------------------------------------
    YOLO_MODEL: str = "yolov8n.pt"                  # n(nano)|s(small)|m(medium) - 정확도 UP -> VRAM UP
    YOLO_DEVICE: str = "cuda"

    # --------------------------------------------------------------
    # AI 모델 - 로컬 LLM ( Qwen2.5-VL-7B, llama-cpp-python)
    # --------------------------------------------------------------
    # Qwen2.5-VL-7B: QLoRA 4bit ~5GB, 12GB VRAM 내 안정 "*Q5_K_M*" --local-dir ./models/llm/
    # 다운로드: hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF --include 
    LLM_MODEL_PATH: str = "./models/llm/"                           # 로컬 GGUF 모델 파일 경로
    LLM_MODEL_NAME: str = "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf"      # GGUF 파일명
    LLM_N_GPU_LAYERS: int = -1                                      # -1: 모든 레이어를 GPU에 오프로드 (12GB VRAM 충분)
    LLM_CTX_SIZE: int = 8192                                        # 하이라이트 추출에 8K 컨텍스트면 충분

    # --------------------------------------------------------------
    # VLM 서버 - llama-server 서브프로세스 (Qwen2.5-VL 멀티모달 추론)
    # --------------------------------------------------------------
    # 서브프로세스 방식: VRAM 완전 해제, OpenAI 호환 API 재활용
    LLAMA_SERVER_PATH: str = "./bin/llama-server"                               # 빌드된 바이너리 경로
    MMPROJ_MODEL_NAME: str = "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"           # 멀티모달 프로젝터 파일명
    LLM_SERVER_HOST: str = "127.0.0.1"                                          # 로컬 전용 (외부 노출 금지)
    LLM_SERVER_PORT: int = 8090                                                 # FastAPI(8000)와 충돌 방지
    LLM_SERVER_TIMEOUT: int = 60                                                # 서버 시작 대기 타임아웃 (초)

    # --------------------------------------------------------------
    # 프레임 추출 — VLM 멀티모달 입력용
    # --------------------------------------------------------------
    # Qwen2.5-VL 비주얼 토큰 예산:
    #   560px -> 프레임당 ~200~300토큰 -> 20프레임 = ~4,000~6,000토큰
    FRAME_EXTRACT_INTERVAL_SEC: float = 10.0
    FRAME_EXTRACT_MAX_FRAMES: int = 20
    FRAME_EXTRACT_RESOLUTION: int = 560

    # --------------------------------------------------------------
    # Phase 2 - 클립 기반 학습 데이터 (10초 클립 + Whisper 전사 텍스트)
    # --------------------------------------------------------------
    # 336px / 1fps -> 프레임당 ~120토큰 -> 10프레임 = ~1,200토큰 (VRAM 절약)
    P2_CLIP_DURATION_SEC: float = 10.0              # 클립 단위 길이 (초)
    P2_FRAME_INTERVAL_SEC: float = 1.0              # 프레임 추출 간격 (1fps)
    P2_FRAME_RESOLUTION: int = 336                  # 프레임 해상도 (px)
    P2_MAX_FRAMES_PER_CLIP: int = 10                # 클립당 최대 프레임 수
    P2_WHISPER_TEXT:bool = True                     # Whisper 전사 텍스트 메타데이터 추가

    # --------------------------------------------------------------
    # 청크 분할 (17일차 추가) - 장편 영상 대응
    # --------------------------------------------------------------
    #
    # 배경:
    #   93분 영상 처리 시 프롬프트가 dir 47K 토큰으로 부풀어 LLM_CTX_SIZE=8192 초과
    #   컨텍스트를 48k로 올리면 KV 캐시 VRAM 스필오버 -> PCIe 병목 발생
    # 해결:
    #   전사를 CHUNK_DURATION_SEC 단위로 분할 -> 각 청크별 LLM 호출 -> 재랭킹으로 병합
    #
    # CHUNK_DURATION_SEC (600초):
    #   10분 청크의 프롬프트 예상 토큰은 약 5K -> 기본 8K 컨텍스트로 안전 수용
    # CHUNK_OVERLAP_SEC (30초):
    #   청크 경계에 걸친 발화 단략 유실 방지 (앞뒤 각 15초)
    # HIGHLIGHT_IOU_THRESHOLD (0.3):
    #   여러 청크에서 같은 구간이 중복 선정된 경우 IoU 30% 이상이면 겹침으로 판단
    CHUNK_DURATION_SEC: float = 600.0
    CHUNK_OVERLAP_SEC: float = 30.0
    HIGHLIGHT_IOU_THRESHOLD: float = 0.3

    # --------------------------------------------------------------
    # 히트맵 수집 (20일차 추가) - YouTube "Most Replayed" 크롤러
    # --------------------------------------------------------------
    # yt-dlp info.heatmap(InnerTube)으로 히트맵 추출 -> 파인튜닝 라벨로 활용
    HEATMAP_OUTPUT_DIR: str = "./data/heatmaps"     # JSONL 저장 경로
    HEATMAP_RATE_LIMIT_SEC: float = 2.0             # 영상 간 대기 (IP 차단 회피)
    HEATMAP_MIN_DURATION_SEC: float = 60.0          # 최소 영상 길이 필터
    HEATMAP_REQUEST_TIMEOUT_SEC: int = 30           # yt-dlp 타임 아웃

    # --------------------------------------------------------------
    # 파인튜닝 데이터 분비 (21일차 추가) - VLM 멀티모달 학습
    # --------------------------------------------------------------
    # 히트맵 피크 구간의 프레임을 추출하여 포지티브/네거티브 학습 데이터 생성
    # 산출물(dataset.jsonl) -> 22일차 Unsolth QLoRA 파인튜닝 입력
    FINETUNE_OUTPUT_DIR: str= "./data/finetune"     # 데이터셋 + 프레임 저장 루트
    FINETUNE_FRAMES_PER_SEGMENT: int = 5            # 세그먼트당 추출 프레임 수
    FINETUNE_NEGATIVE_RATIO: float = 1.0            # 네거티브/포지티브 비율 (1:1)
    FINETUNE_MIN_PEAK_COUNT: int = 2                # 처리 대상 최소 피크 수

    # --------------------------------------------------------------
    # QLoRA 파인튜닝 (22일차 추가) - Unsloth LoRA 어댑터
    # --------------------------------------------------------------
    # Qwen2.5-VL-7B 4bit + QLoRA: ~5GB 베이스 + 학습 오버헤드 -> 12GB 내 안정
    # 학습 완료 후 adapter/ 디렉토리에 LoRA 가중치 저장 (~100MB)
    LORA_BASE_MODEL: str = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"            # Unsloth 4bit 사전 양자화
    LORA_OUTPUT_DIR: str ="./models/lora/heatmap_adapter"                               # 판별기 어댑터 경로
    LORA_GENERATOR_DIR: str = "./models/lora/heatmap_generator"                         # 생성기 어댑터 경로
    LORA_ENABLED: bool = False                                                          # True면 추론 시 LoRA 어댑터 로드
    # 33일차: Phase 1/2 품질 비교 테스트용 파이프라인 전환
    LORA_PIPELINE: str = "phase2"                                                       # "phase2"(회귀 슬라이딩 윈도우) | "phase1"(생성기 단독 1회 추론)
    LORA_PHASE1_ADAPTER: str = "adapter_backup_0246"                                    # Phase 1 생성기 어댑터 디렉토리명 (LORA_GENERATOR_DIR 하위)
    LORA_PHASE1_VERIFY: bool = False                                                    # True면 Phase 1 후보를 판별기로 추가 검증 (테스트 b)

    # --------------------------------------------------------------
    # 탐색 샘플링 (36일차 추가) - Component F: top-K 활용 외 피드백 다양화
    # --------------------------------------------------------------
    EXPLORATION_COUNT: int = 1                      # 영상당 탐색 슬롯 수 (0이면 F 비활성=기존 동작)
    EXPLORATION_MIN_SCORE: float = 0.2              # 탐색 후보 최저 점수 (0점대 쓰레기 윈도우 차단)

    # --------------------------------------------------------------
    # 외부 API (선택 사항)
    # --------------------------------------------------------------
    # 빈 문자열이면 로컬 llm 사용, 값이 있으면 API 사용
    OPENAI_API_KEY: str = ""

    # --------------------------------------------------------------
    # 영상 처리
    # --------------------------------------------------------------
    OUTPUT_DIR: str = "./outputs"                   # 완성된 쇼츠 영상 저장 경로
    TEMP_DIR: str = "./temp"                        # 임시 작업 파일 경로 (다운로드, 중간 결과물)
    MAX_VIDEO_DURATION_MIN: int = 120               # 처리 가능한 최대 영상 길이 (분)
    DEFAULT_SHORTS_DURATION_SEC: int = 60           # 기본 쇼츠 길이 (초)
    VIDEO_QUALITY: int = 1080                       # 다운로드 해상도 (720 또는 1080)

    # --------------------------------------------------------------
    # FFmpeg (GPU 가속 인코딩)
    # --------------------------------------------------------------
    FFMPEG_HWACCEL: str = "cuda"                    # GPU 가속 방식 (cuda 또는 none)
    NVENC_PRESET: str = "p4"                        # p1(최고속) ~ p7(최고품질) - p4는 속도/품질 균형
    NVENC_CQ: int = 23                              # 품질 지수 (낮을 수록 고품질, 18~28 권장)

    # --------------------------------------------------------------
    # 로깅
    # --------------------------------------------------------------
    LOG_LEVEL: str = "DEBUG"                        # DEBUG|INFO|WARNING|ERROR
    LOG_FILE: str = "./logs/app.log"                # 로그 파일 저장 경로

    # --------------------------------------------------------------
    # 편의 프로퍼티 (계산된 값)
    # --------------------------------------------------------------
    @property
    def is_dev(self) -> bool:
        """현재 개발 환경인지 판별. DB 쿼리 로그 출력 등에 사용"""

        return self.APP_ENV == "development"
    
    @property
    def output_path(self) -> Path:
        """출력 디렉토리 Path 객체, 존재하지 않으면 자동 생성"""

        p = Path(self.OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def temp_path(self) -> Path:
        """임시 디렉토리 Path 객체, 존재하지 않으면 자동 생성"""
        
        p = Path(self.TEMP_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def llm_model_file(self) -> Path:
        """LLM GGUF 모델 전체 경로 (파일이면 그대로, 디렉토리면 + MODEL_NAME)"""
        
        path = Path(self.LLM_MODEL_PATH)
        if path.is_file() and path.suffix == ".gguf":
            return path      
        return path / self.LLM_MODEL_NAME
    
    @property
    def mmproj_model_file(self) -> Path:
        """멀티모달 프로젝터(mmproj) 파일의 전체 경로 (LLM_MODEL_PATH + MMPROJ_MODEL_NAME)"""

        return Path(self.LLM_MODEL_PATH) / self.MMPROJ_MODEL_NAME
    
    @property
    def heatmap_output_path(self) -> Path:
        """히트맵 JSONL 출력 디렉토리 (자동 생성)"""

        p = Path(self.HEATMAP_OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def finetune_output_path(self) -> Path:
        """파인튜닝 데이터셋 디렉토리 (frames/ 서브디렉토리 포함 자동 생성)"""

        p = Path(self.FINETUNE_OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def lora_adapter_path(self) -> Path:
        """판별기 LoRA 어댑터 경로"""

        return Path(self.LORA_OUTPUT_DIR) / "adapter"
    
    @property
    def lora_generator_path(self) -> Path:
        """생성기 LoRA 어댑터 경로"""

        return Path(self.LORA_GENERATOR_DIR) / "adapter"
    
    @property
    def lora_phase1_path(self) -> Path:
        """33일차: Phase 1 생성기 어댑터 경로 (품질 비교 테스트용)"""

        return Path(self.LORA_GENERATOR_DIR) / self.LORA_PHASE1_ADAPTER
    
# 최초 1회만 Settings 객체 생성, 이후 호출에서는 캐시된 동일 객체 반환
@lru_cache
def get_settings() -> Settings:
    """
    설정 싱글턴 팩토리

    DI(의존성 주입)에서 사용:
        settings = Depends(get_settings)
    
    @lru_cache 덕분에 앱 전체에서 동일한 Settings 인스턴스를 공유
    """
    return Settings()

# 모든 레벨 변수: 다른 모델에서 from app.core.config import settings 로 바로 사용
settings = get_settings()