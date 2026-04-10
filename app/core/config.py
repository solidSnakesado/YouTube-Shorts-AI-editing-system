# 계층: 인프라 계층 (Core)
# 역할: 애플리케이션 전역 환경 설정을 중앙 관리
#       .env 파일의 값을 읽어와 Python 타입으로 자동 변환하며,
#       잘못된 값이 들어오면 서버 시작 시점에 즉시 에러를 발생 시킨다.
# 의존: 없음 (최하의 인프라 모듈 - 다른 모듈이 이 모듈에 의좀)
# MVA 원칙: 설정 외부화 - 환결 변수 하드코딩 금지, 환경별 설정 분리

"""
환경 설정 모듈

모든 환경 변수를 Pydantic Setting로 관리한다.
.env 파일에서 값을 읽어오며, 타입 검증이 자동으로 수핸된다.
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

    타입 힌트(str, int, bool 등)에 따라 자동 변횐되므로,
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
    APP_NAME: str = "YT Shorts AI"                                              # Swagger UI 제모ㅓㄱ에 표시
    APP_ENV: Literal["development", "staging", "production"] = "development"    # Literal 타입: 이 세 값 중 하나만 허용, 그 외 값이 들어오면 서버 시작시 ValidationError 발생
    APP_DEBUG: bool = True                                                      # True 이면 SQLAlchemy 쿼리 로그 출력 등 디버그 모드 활성화
    APP_HOST: str = "0.0.0.0"                                                   # 0.0.0.0 = 모든 네크워크 인터페이스에서 접속 허용
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
    # AI 모델 설정
    # --------------------------------------------------------------
    WHISPER_MODEL_SIZE: str = "medium"              # tiny|base|small|medium|lagge-v3-torbo
    WHISPER_DEVICE: str = "cuda"                    # cuda(GPU) 또는 cpu
    WHISPER_COMPUTE_TYPE: str = "float16"           # float16(빠름) 또는 int8(VRAM 절약)

    YOLO_MODEL: str = "yolov8n.pt"                  # n(nano)|s(small)|m(medium) - 정확도 UP -> VRAM UP
    YOLO_DEVICE: str = "cuda"

    LLM_MODEL_PATH: str = "./models/llm/"           # 로컬 GGUF 모델 파일 경로
    LLM_N_GPU_LAYERS: int = 35                      # GPU에 오프로드할 레이어 수 (-1 -> 전부)
    LLM_CTX_SIZE: int = 4096                        # LLM 컨텍스트 윈도우 크기 (토큰)

    # --------------------------------------------------------------
    # 외부 API (선택 사항)
    # --------------------------------------------------------------
    # 빈 문자열이면 로컬 LLM 사용, 값이 있으면 API 사용
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
        """
        현재 개발 환경인지 판별. DB 쿼리 로그 출력 등에 사용
        """
        return self.APP_ENV == "development"
    
    @property
    def output_path(self) -> Path:
        """
        출력 디렉토리 Path 객체, 존재하지 않으면 자동 생성
        """
        p = Path(self.OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def temp_path(self) -> Path:
        """
        임시 디렉토리 Path 객체, 존재하지 않으면 자동 생성
        """
        p = Path(self.TEMP_DIR)
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