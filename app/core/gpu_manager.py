# 계층: 인프라 계층 (Core)
# 역할: GPU VRAM 모델 로드/언로드/스위칭 전략을 중앙 관리
#       Whisper, LLM, YOLO 등 GPU 모델의 생명주기를 통합 제어하여
#       한정된 VRAM(11.9GB)에서 모델 간 안전한 교체를 보장
# 의존: app.core.config (모델 설정값)
# MVA 원칙: GPU 리소스 관리는 인프라 책임, 서비스 계층에서 분리
#
# 사용처:
#   - AnalysisService: Whisper 로드/언로드(3~5일차)
#   - AnalysisService: LLM 로드/언로드(6~7일차)
#   - EditingService: YOLO 로드/언로드(8~10일차)
#
# 6~7일차 변경사항:
#   - load_llm() 추가: OpenAI API / 로컬 Gemma 4 GGUF 이중 경로
#   - _resolve_gguf_path() 추가: 모델 파일 탐색 + 다운로드 안내
#
# 8~10일차 변경사항:
#   - load_yolo() 추가: YOLOv8n GPU 로드 (리프레이밍용 객체 탐지)

"""
GPU 모델 관리자

VRAM 모델 로드/언로드 및 메모리 해제를 중앙 관리
모델 스위칭 전략의 핵심 모듈
"""

import gc                               # 가비지 컬렉터: 순환 참조 포함 객체 해제   
from pathlib import Path                # OS 독립적 파일 경로 처리
from typing import Any, Optional        

from loguru import logger               # 구조화된 로깅 라이브러리

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
        1. gc.collect(): 순환 참조 포함 Python 객체 해제
        2. torch.cuda.empty_cache(): PyTorch VRAM 캐시 반환
    """

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass                            # torch 미설치 환경에서도 에러 없이 진행

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

    # 지연 임포트 faster_whisper가 설치되지 않은 환경에서 다른 기능 사용 가능
    from faster_whisper import WhisperModel

    logger.info(
        f"Whisper 모델 로드 시작 | 모델: {settings.WHISPER_MODEL_SIZE} | "
        f"디바이스: {settings.WHISPER_DEVICE} | compute_type: {settings.WHISPER_COMPUTE_TYPE}"
    ) 

    model = WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )

    logger.info(f"Whisper 모델 로드 완료: {settings.WHISPER_MODEL_SIZE}")
    return model

def load_llm() -> dict:
    """
    LLM 모델 로드 (이중 경로) - 6~7일차 신규

    경로 분기:
        OPENAI_API_KEY 있음 -> OpenAI 클라이언트 반환 (GPU 미사용)
        OPENAI_API_KEY 없음 -> Gemma 4 E4B Q8_0 GGUF 로컬 로드

    Gemma 4 E4B Q8_0 선택 이유
        - 파라미터: 4B effective (PLE 아키텍처로 파라미터 효율 극대화)
        - VRAM: ~8~9GB -> RTX 5070 Ti (11.9GB)에 여유롭게 적재
        - 컨텍스트: 최대 128K 지원 (설정에서 8K 사용)
        - 26B-A4B(UD-Q4_K_XL)는 12GB에서 fit-based 배치 필요 + OOM 위험 -> 안전성을 위해 E4B Q8_0 선택
        - llama.cpp에서 Gemma 4 GGUF 공식 지원 확인 완료 (2026.04 기준)

    다운로드 명령어:
        huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \\
            --include "*Q8_0*" --local-dir ./model/llm/

    Returns:
        {"type": "openai", "client": OpenAI()} 또는
        {"type": "local", "model": Llama()}
    """

    # 경로 1: OpenAI API (VRAM 미사용, 클라우드)
    if settings.OPENAI_API_KEY:
        logger.info("LLM 로드: OpenAI API 모드 (GPU 미사용)")
        # 지연 임포트: openai 미설치 시 로컬 경로만 사용 가능
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return {"type": "openai", "client": client}
    
    # 경로 2: 로컬 Gemma 4 E4B GGUF (GPU 사용)
    model_path = _resolve_gguf_path()

    logger.info(
        f"LLM 로드 시작 | Gemma 4 E4B Q8_0 | "
        f"파일: {model_path.name} | "
        f"GPU 레이어: {settings.LLM_N_GPU_LAYERS} | "
        f"컨텍스트: {settings.LLM_CTX_SIZE}" 
    )

    # 지연 임포트: llama-cpp-python이 설치되지 않은 환경 대응
    from llama_cpp import Llama

    # n_gpu_layers=-1: 모든 레이어를 GPU에 올림 (12GB VRAM이면 E4B 전체 적재 가능)
    # n_ctx=8192: 하이라이트 추출에 필요한 컨텍스트 크기 (전사 텍스트 + 프롬프트)
    # verbose=False: llama.cpp 내부 로그 비활성화 (loguru로 통합)
    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=settings.LLM_N_GPU_LAYERS,
        n_ctx=settings.LLM_CTX_SIZE,
        verbose=False,
    )

    vram = get_vram_status()
    logger.info(
        f"LLM 로드 완료: {model_path.name} | "
        f"VRAM: {vram['allocated_mb']}MB"
    )
    return {"type": "local", "model": model}

def load_yolo() -> Any:
    """
    YOLOv8 객체 탐지 모델 로드 - 8~10일차 신규

    리프레이밍에서 프레임별 피사체(인물) 위치 추적에 사용

    모델 크기별 VRAM (RTX 5070 Ti 11.9GB):
        YOLOv8n (nano):     ~1~2GB VRAM, ~200 FPS (채택)
        YOLOv8s (small):    ~2~3GB VRAM, ~150 FPS
        YOLOv8m (medium):   ~4~5GB VRAM, ~100 FPS

    Returns:
        ultralytics.YOLO 인스턴스 (GPU에 로드됨)
    """

    # 지연 임프트: ultralytics 미설치 환경에서 다른 기능 사용 가능
    from ultralytics import YOLO

    model_name = settings.YOLO_MODEL        # 기본값: "yolov8n.pt"
    device = settings.YOLO_DEVICE           # 기본값: "cuda"

    logger.info(f"YOLO 모델 로드 시작 | 모델: {model_name} | 디바이스: {device}")

    # YOLO()는 모델 파일이 없으면 자동 다운로드 (ultralytics 내장)
    # model.yolo/ 디렉토리에 캐싱됨
    model = YOLO(model_name)

    # GPU로 이동 (cuda/cpu 분기)
    # model.to()는 내부적으로 PyTorch .to(device) 호출
    model.to(device)

    vram = get_vram_status()
    logger.info(
        f"YOLO 모델 로드 완료: {model_name} | "
        f"VRAM: {vram['allocated_mb']}MB"
    )
    return model

def _resolve_gguf_path() -> Path:
    """
    GGUF 모델 파일 경로 확정 - 6~7일차 신규

    탐색 순서:
        1. settings.llm_model_file - config에서 조합된 전체 경로
           (LLM_MODEL_PATH + LLM_MODEL_NAME)
        2. LLM_MODEL_PATH 디렉토리 내 .gguf 파일 자동 탐색
           (여러 개면 가장 큰 파일 선택 - 보통 더 정확한 양자화)
        3. 못 찾으면 FileNotFoundError + 다운로드 안내 메시지

    Returns:
        확정된 GGUF 파일 Path
    """

    # 1. config에서 지정한 경로 확인 (LLM_MODEL_PATH + LLM_MODEL_NAME)
    configured = settings.llm_model_file
    if configured.is_file():
        return configured
    
    # 2. 디렉토리 내 .gguf 파일 자동 탐색
    model_dir = Path(settings.LLM_MODEL_PATH)
    if model_dir.is_dir():
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            # 가장 큰 파일 선택 (Q8_0 > Q4_K_M > Q3_K 순으로 보통 파일 크기가 큼)
            chosen = max(gguf_files, key=lambda f: f.stat().st_size)
            logger.info(f"GGUF 자동 탐색: {chosen.name}")
            return chosen
        
    # 3. 못 찾음 -> 다운로드 안내
    raise FileNotFoundError(
        f"GGUF 모델 파일을 찾을 수 없습니다.\n"
        f"  설정 경로: {configured}\n"
        f"  탐색 디렉토리: {model_dir}\n\n"
        f"다운로드 방법:\n"
        f"  huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \\\n"
        f"      --include '*Q8_0*' --local-dir ./models/llm/\n\n"
        f"또는 OPENAI_API_KEY를 .env에 설정하세요."
    )

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
    
    del model                   # 참조 제거 -> GC 대상으로 등록
    release_vram()              # gc.collect() + torch.cuda.empty_cache()

    vram = get_vram_status()
    logger.info(
        f"{model_name} 언로드 완료 | "
        f"VRAM allocated: {vram['allocated_mb']}MB / VRAM reserved: {vram['reserved_mb']}MB"
    )