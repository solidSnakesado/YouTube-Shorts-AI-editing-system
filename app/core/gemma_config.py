# 계층: 설정 계층 (Pydantic Settings)
# 역할: Gemma 4 E4B 오디오 피벗 전용 설정 - 기존 config.Settings(Qwen)와 격리
# 39일차 신규: Qwen 유지 + Gemma 추가(모델 셀렉터) 토대. config.py 무수정
#   - 동일 .env를 읽되 GEMMA_* 키만 사용(extra=ignore) -> 기존 Settings와 충돌 없음
#   - 데이터 재구축(1fps 프레임 + 30s 오디오) + 추론 셀렉터(베이스/어댑터/활성) 공통 토대

"""Gemma 4 E4B 오디오 피벗 전용 설정 (기존 Qwen config와 물리적 격리)"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class GemmaSettings(BaseSettings):
    """Gemma 4 E4B 오디오 피벗 전용 설정
    
    기존 config.Settings(Qwen 스택)와 분리된 별도 객체. 동일한 .env를 읽지만
    GEMMA_* 필드만 사용하고 모르는 키는 무시(extra=ignore)하므로 기존 Settings와
    충돌하지 않는다. config.py는 일절 수정하지 않는다."""

    # 39일차: 기존 Settings와 동일 규칙(.env/utf-8/case_sensitive) + extra=ignore
    #   extra=ignore -> .env의 비-GEMMA 키(LORA_* 등)를 무시하여 기존 Settings와 공존
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --------------------------------------------------------------
    # Gemma 4 E4B 모델 (39일차) - 학습=Colab A100 / 추론=로컬 12GB
    # --------------------------------------------------------------
    # 공식 Unsloth 모델 문자열, 로컬 추론 FastModel 베이스 + Colab 학습 베이스 공통
    GEMMA_BASE_MODEL: str = "unsloth/gemma-4-E4B-it"
    # Qwen LORA_GENERATOR_DIR과 분리된 별도 어댑터 디렉토리 (격리)
    GEMMA_GENERATOR_DIR: str = "./models/lora/gemma_generator"
    # True면 셀렉터에 Gemma 노출. 어댑터 학습 완료 후 .env에서 활성화
    GEMMA_ENABLED: bool = False

    # --------------------------------------------------------------
    # 데이터 상한 (39일차) - 피벗 스펙: [1fps 프레임 + 30s 오디오, 정렬]
    # --------------------------------------------------------------
    # 39일차: 학습 클립 윈도우 = 오디오와 동일 30s (정렬) -> 1fps x 30 = 30프레임 + 30s 오디오
    GEMMA_AUDIO_MAX_SEC: int = 30           # 오디오 + 학습 클립 윈도우 공통 (Gemma 네이티브 30s, 둘이 정렬)
    GEMMA_FRAME_FPS: int = 1                # 프레임 추출 주기 (1fps)
    GEMMA_FRAME_RESOLUTION: int = 512       # 프레임 해상도 (짧은 변, scale={res}:-1). 클수록 화질/토큰/저장 증가 (조정 가능)
    GEMMA_OUTPUT_MAX_SEC: int = 60          # 출력 쇼츠 길이 상한 (추론 시 탐지 윈도우 확장용, 데이터 빌드 미사용)

    @property
    def gemma_generator_path(self) -> Path:
        """Gemma 생성기 어댑터 경로 (config.lora_generator_path와 동일 규칙: DIR/adapter)"""

        return Path(self.GEMMA_GENERATOR_DIR) / "adapter"
    
@lru_cache
def get_gemma_settings() -> GemmaSettings:
    """Gemma 설정 싱글턴 팩토리 (config.get_settings와 동일 패턴)"""

    return GemmaSettings()

# 모듈 레벨: from app.core.gemma_config import gemma_settings 로 바로 사용
gemma_settings = get_gemma_settings()