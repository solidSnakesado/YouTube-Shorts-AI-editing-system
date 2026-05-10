# 계층: 인프라 계층 (Core)
# 역할: 애플리케이션 전역 환경 설정을 중앙 관리
#       .env 파일의 값을 읽어와 Python 타입으로 자동 변환하며,
#       잘못된 값이 들어오면 서버 시작 시점에 즉시 에러를 발생 시킨다.
# 의존: 없음 (최하위 인프라 모듈 - 다른 모듈이 이 모듈에 의존)
# MVA 원칙: 설정 외부화 - 환경 변수 하드코딩 금지, 환경별 설정 분리
#
# 6~7일차 변경사항:
#   - LLM_MODEL_NAME 필드 추가 (GGUF 파일명 명시)
#   - LLM_N_GPU_LAYERS 기본값 -1로 변경 (전체 GPU 오프로드)
#   - LLM_CTX_SIZE 기본값 8192로 변경 (하이라이트 추출에 충분)
#   - llm_model_file 프로퍼티 추가 (경로 + 파일명 조합)
#
# 14일차 변경사항:
#   - VLM 서버 설정 섹션 추가 (LLAMA_SERVER_PATH, LLM_SERVER_PORT 등)
#   - MMPROJ_MODEL_NAME 필드 추가 (멀티모달 프로젝터 파일명)
#   - 프레임 추출 설정 섹션 추가 (FRAME_EXTRACT_* 3개)
#   - mmproj_model_file 프로퍼티 추가
#
# 17일차 변경사항:
#   - 청크 분할 설정 3개 추가 (CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC, HIGHLIGHT_IOU_THRESHOLD)
#   - 장편 영상 대응을 위한 전사 청크 분할 기능에 사용됨
#   - llm_model_file 프로퍼터 추가 (경로 + 파일명 조합)
#
# 20일차 변경사항:
#   - 히트맵 수집 설정 4개 추가 (HEATMAP_OUTPUT_DIR, HEATMAP_RATE_LIMIT_SEC,
#     HEATMAP_MIN_DURATION_SEC, HEATMAP_REQUEST_TIMEOUT_SEC)
#   - YouTube "Most Replayed" 히트맵 크롤러용 (파인튜닝 데이터 수집)
#   - heatmap_output_path 프로퍼티 추가 (디렉토리 자동 생성)

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
    # AI 모델 - 로컬 LLM (Gemma 4 E4B, llama-cpp-python)
    # --------------------------------------------------------------
    # Gemma 4 E4B Q8_0: ~8~9GB VRAM, RTX 5070 Ti (11.9GB)에 적합
    # 모델 다운로드 명령어
    #   huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    #     --include "*Q8_0*" --local-dir ./model/llm/
    LLM_MODEL_PATH: str = "./models/llm/"               # 로컬 GGUF 모델 파일 경로
    LLM_MODEL_NAME: str = "gemma-4-E4B-it-Q8_0.gguf"    # GGUF 파일명 (6~7일차 추가)
    LLM_N_GPU_LAYERS: int = -1                          # -1: 모든 레이어를 GPU에 오프로드 (12GB VRAM 충분)
    LLM_CTX_SIZE: int = 8192                            # 하이라이트 추출에 8K 컨텍스트면 충분

    # --------------------------------------------------------------
    # VLM 서버 - llama-server 서브프로세스 (14일차 추가)
    # --------------------------------------------------------------
    # llama-server 네이티브 바이너리로 Gemma 4 멀티모달(텍스트 + 이미지) 추론
    # llama-cpp-python 대신 서브프로세스 방식 채택 이유:
    #   - llama.cpp 최신 libmtmd 멀티모달 지원 활용
    #   - 프로세스 종료 시 VRAM이 OS 수준에서 완전 해제
    #   - OpenAI 호환 API(/v1/chat/completions) -> 기존 코드 재활용 가능
    LLAMA_SERVER_PATH: str = "./bin/llama-server"       # 빌드된 바이너리 경로
    MMPROJ_MODEL_NAME: str = "mmproj-BF16.gguf"         # 멀티모달 프로젝터 파일명 (~800MB)
    LLM_SERVER_HOST: str = "127.0.0.1"                  # 로컬 전용 (외부 노출 금지)
    LLM_SERVER_PORT: int = 8090                         # FastAPI(8000)와 충돌 방지
    LLM_SERVER_TIMEOUT: int = 60                        # 서버 시작 대기 타임아웃 (초)

    # --------------------------------------------------------------
    # 프레임 추출 — VLM 멀티모달 입력용 (14일차 추가)
    # --------------------------------------------------------------
    # Gemma 4 비주얼 토큰 예산:
    #   560px -> 프레임당 ~280토큰 -> 20프레임 = ~5,600토큰 (일반 분석)
    #   1120px -> 프레임당 ~1,120토큰 -> 10프레임 = ~11,200토큰 (OCR/세부)
    FRAME_EXTRACT_INTERVAL_SEC: float = 10.0
    FRAME_EXTRACT_MAX_FRAMES: int = 20
    FRAME_EXTRACT_RESOLUTION: int = 560

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
    #
    # 목적:
    #   YouTube의 "Most Replayed" 히트맵 데이터를 다수 영상에서 수집하여 
    #   21일차 멀티모달 파인튜닝 데이터셋의 라벨로 사용
    #
    # 동작 메커니즘:
    #   YouTube Data API는 히트맵을 공식 노출하지 않으므로 yt-dlp의
    #   info.heatmap 필드(InnerTube에서 파싱)를 통해 추출
    #
    # HEATMAP_OUTPUT_DIR:
    #   수집 결과 JSONL 파일 저장 경로 (영상별 1라인)
    # HEATMAP_RATE_LIMIT_SEC:
    #   영상 간 호출 간격 (초), YouTube IP 차단 회피용 매너모드
    # HEATMAP_MIN_DURATION_SEC:
    #   너무 짧은 영상은 히트맵이 없거나 노이즈가 심함 - 필터 기준
    # HEATMAP_REQUEST_TIMEOUT_SEC:
    #   yt-dlp metadata 호출 타임아웃 (네트워크 지연 시 hang 방지)
    HEATMAP_OUTPUT_DIR: str = "./data/heatmaps"
    HEATMAP_RATE_LIMIT_SEC: float = 2.0
    HEATMAP_MIN_DURATION_SEC: float = 60.0
    HEATMAP_REQUEST_TIMEOUT_SEC: int = 30

    # --------------------------------------------------------------
    # 외부 API (선택 사항)
    # --------------------------------------------------------------
    # 빈 문자열이면 로컬 Gemma 4 사용, 값이 있으면 API 사용
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
        """
        LLM GGUF 모델 파일의 전체 경로 (6~7일차 추가)

        LLM_MODEL_PATH가 직접 .gguf 파일이면 그대로 반환
        디렉토리이면 LLM_MODEL_NAME과 결합하여 반환

        ex> "./models/llm/" + "gemma-4-E4B-it-Q8_0.gguf"
            -> Path("./models/llm/gemma-4-E4B-it-Q8_0.gguf")
        """
        
        path = Path(self.LLM_MODEL_PATH)
        if path.is_file() and path.suffix == ".gguf":
            return path
        
        return path / self.LLM_MODEL_NAME
    
    @property
    def mmproj_model_file(self) -> Path:
        """
        멀티모달 프로젝터(mmproj) 파일의 전체 경로 (14일차 추가)
        LLM_MODEL_PATH 디렉토리 + MMPROJ_MODEL_NAME 조합
        ex> "./models/llm/" + "mmproj-BF16.gguf" -> Path("./models/llm/mmproj-BF16.gguf")
        """

        return Path(self.LLM_MODEL_PATH) / self.MMPROJ_MODEL_NAME
    
    @property
    def heatmap_output_path(self) -> Path:
        """
        히트맵 JSONL 출력 디렉토리 (20일차 추가)

        다수 영상의 히트맵을 누적 저장하는 디렉토리
        존재하지 않으면 자동 생성 (parents=True로 상위 디렉토리도 같이 생성)

        ex> ./data/heatmaps/heatmaps_2026-05-07.jsonl
        """

        p = Path(self.HEATMAP_OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)

        return p
    
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