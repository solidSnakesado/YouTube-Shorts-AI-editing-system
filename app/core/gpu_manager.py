내용 추가 필요
# 계층: 인프라 계층 (Core)
# 역할: GPU VRAM 모델 로드/언로드/스위칭 전략을 중앙 관리
#       Whiper, LLM, YOLO 등 GPU 모델의 생명주기를 통합 제어하여
#       한정된 VRAM(11.9GB)에서 모델 간 안전한 교체를 보장
# 의존: app.core.config (모델 설정값)
# MVA 원칙: GPU 리소스 관리는 인프라 책임, 서비스 계층에서 분리
#
# 사용처:
#   - AnalysisService: Whisper 로드/언로드(3~5일차)
#   - AnalysisService: LLM 로드/언로드(6~7일차)
#   - EditingService: YOLO 로드/언로드(8~10일차)

"""
GPU 모델 관리자

VRAM 모델 로드/언로드 및 메모리 해제를 중앙 관리
모델 스위칭 전략의 핵심 모듈
"""

import gc
from typing import Any, Optional

from loguru import logger

from app.core.config import settings

def release_vram() -> None:
    """
    GPU VRAM 캐시 메모리 강제 해제

    Python GC + CUDA 캐시 해제를 수행하여
    이전 모델이 점유한 VRAM을 OS 수준에서 반환

    호출 시점:
        - 모델 언로드 직후
        - 다음 모델 모드 직전 (안전 장치)

    해제 순서:
        1. gc,collect(): 순환 참조 포함 Python 객체 해제
        2. torch.cuda.empty_cache(): PyTorch VRAM 캐시 반환
    """

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def get_vram_status() -> dict:
    """
    현재 GPU VRAM 사용 상태 조회

    Returns:
        {"allocated_mb": 1234, "reserved_mb": 2048, "available": True}
        torch 미설치 시: {"allocated_mb": 0, "reserved_mb": 0, "available": False}
    """

    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
                "available": True,
            }
    except ImportError:
        pass

    return {"allocated_mb": 0, "reserved_mb": 0, "available": False}
    
def load_whisper() -> Any:
    """
    faster-whisper 모델 로드

    settings의 WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE을 사용

    모델 크기별 VRAM (RTX 5070 Ti 11.9GB):
        tiny/base: 1~1.5GB | small: ~2.5GB | medium: ~5GB (기본값)
        large-v3: ~10GB | large-v3-turbo: ~6GB (권장)

    compute_type:
        float16(기본, GPU 최적) | int8(VRAM 절약) | float32(CPU 전용)

    Returns:
        WhisperModel 인스턴스
    """

    from faster_whisper import WhisperModel

    logger.info(
        f"Whisper 모델 로드 시작 | "
        f"모델: {settings.WHISPER_MODEL_SIZE} | "
        f"디바이스: {settings.WHISPER_DEVICE} | "
        f"compute_type: {settings.WHISPER_COMPUTE_TYPE}"
    ) 

    model = WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )

    logger.info(f"Whisper 모델 로드 완료: {settings.WHISPER_MODEL_SIZE}")
    return model

def unload_model(model: Optional[Any], model_name: str = "모델") -> None:
    """
    GPU 모델 언로드 및 VRAM 해제 (범용)

    Whisper, LLM, YOLO 등 어떤 모델이든 동일한 절차로 해제
    del -> gc.collect -> torch.cuda.empty_cache 3단계 수행

    Args:
        model: 언로드할 모델 인스턴스 (None이면 아무 작업 안 함)
        model_name: 로깅용 모델 이름 (ex> "Whisper", "YOLO")
    """

    if model is None:
        return
    
    del model
    release_vram()

    vram = get_vram_status()
    logger.info(
        f"{model_name} 언로드 완료 | "
        f"VRAM allocated: {vram['allocated_mb']}MB | "
        f"VRAM reserved: {vram['reserved_mb']}MB"
    )