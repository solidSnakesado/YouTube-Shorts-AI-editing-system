# 계층: 인프라 계층 (Core)
# 역할: llama-server 서브프로세스의 생명주기(시작/종료/헬스체크) 관리
#       gpu_manager.py의 300줄 규칙 준수를 위해 분리된 인프라 모듈
# 의존: app.core.config (서버 경로, 포트, 타임아웃 설정)
#       app.core.gpu_manager (_resolve_gguf_path: GGUF 모델 파일 탐색)
# MVA 원칙: GPU 리소스 관리는 인프라 책임, 서비스 계층에서 분리
#
# 사용처:
#   - AnalysisService: VLM 멀티모달 분석 시 서버 시작/종료 (15일차 예정)
#
# 14일차 신규:
#   - start_llm_server(): llama-server 서브프로세스 시작 (텍스트/멀티모달)
#   - stop_llm_server(): 서브프로세스 종료 + VRAM OS 수준 해제
#   - _wait_for_health(): /health 엔드포인트 폴링으로 서버 준비 확인
#
# 왜 llm-server 서브프로세스인가?
#   llama.cpp 네이티브 서버는 libmtmd 멀티모달을 안정적으로 지원하며,
#   서브프로세스 종료 시 VRAM이 OS 수준에서 완전 해제되는 이점도 있음.
#   OpenAI 호환 API(/v1/chat/completions)로 통신하므로 기존 코드 재활용 가능.
# 22일차: Qwen2.5-VL-7B 전환에 따른 샘플링 파라미터 및 에러 메시지 업데이트

"""
LLM 서버 관리자 - llama-server 서브프로세스 생명주기 관리

텍스트 전용 또는 멀티모달(이미지 + 텍스트) 모드로 llama-server를 시작하고,
/health 엔드포인트로 준비 상태를 확인한 뒤 서비스 계층에 제공
사용 후 stop_llm_server()로 종료하면 VRAM이 OS 수준에서 완전 해제됨
"""

import subprocess                                       # llama-server 서브프로세스 실행/종료
import time                                             # health check 폴링 대기
from pathlib import Path                                # 바이너리/모델 파일 경로 처리
from typing import Optional
from urllib.request import urlopen                      # health check HTTP 요청 (외부 의존성 없이)
from urllib.error import URLError                       # health check 연결 실패 처리

from loguru import logger                               # 구조화된 로깅

from app.core.config import settings
from app.core.gpu_manager import _resolve_gguf_path     # GGUF 모델 파일 탐색 재활용

# --------------------------------------------------------------
# 서버 시작
# --------------------------------------------------------------

def start_llm_server(multimodal: bool = False) -> subprocess.Popen:
    """
    llama-server를 서브프로세스로 시작

    llama.cpp 네이티브 바이너리를 실행하여 OpenAI 호환 API를 제공
    multimodal=True 시 --mmproj 옵션을 포함하여 이미지 + 텍스트 동시 분석 가능.

    Args:
        multimodal: True -> mmproj 포함 (VLM 모드, 이미지 + 텍스트)
                    False -> 텍스트 전용 모드

    Returns:
        실행 중인 서브프로세스 객체 (stop_llm_server()로 종료)

    Raises:
        FileNotFoundError: llama-server 바이너리 또는 모델 파일 미존재
        TimeoutError: 서버가 타임아웃 내에 준비되지 않음

    VRAM 전략:
        llama-server는 별도 프로세스이므로 PyTorch VRAM과 독립적
        종료 시 OS가 프로세스 메모리를 회수하여 VRAM이 완전 해제됨
        따라서 gc.collect() + torch.cuda.empty_cache() 불필요
    """

    # 바이너리 존재 확인
    server_path = Path(settings.LLAMA_SERVER_PATH)
    if not server_path.is_file():
        raise FileNotFoundError(
            f"llama-server 바이너리를 찾을 수 없습니다: {server_path}\n"
            f"빌드 방법:\n"
            f"  git clone https://github.com/ggml-org/llama-cpp\n"
            f"  cmake llama.cpp -B llama.cpp/build "
            f"-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON "
            f"-DCMAKE_CUDA_ARCHITECTURES=120\n"
            f"  cmake --build llama.cpp/build --config Release "
            f"-j --clean-first --target llama-server\n"
            f"  cp llama.cpp/build/bin/llama-server {server_path}"
        )
    
    # GGUF 모델 파일 경로 확정 (gpu_manager의 기존 탐색 로직 재활용)
    model_path = _resolve_gguf_path()

    # 기본 실행 명령어 - Qwen2.5-VL-7B 범용 샘플링 파라미터
    cmd = [
        str(server_path),
        "-m", str(model_path),
        "--port", str(settings.LLM_SERVER_PORT),
        "--host", settings.LLM_SERVER_HOST,
        "--temp", "0.3",                            # 하이라이트 추출: 정확성 우선
        "--top-p", "0.95",                          
        "--ctx-size", str(settings.LLM_CTX_SIZE),
        "--batch-size", "1024",
        "--ubatch-size", "1024",
        "-ngl", str(settings.LLM_N_GPU_LAYERS),
    ]

    # 멀티모달 모드: mmproj 프로젝터 + 이미지 토큰 제한 추가
    if multimodal:
        mmproj_path = settings.mmproj_model_file
        if not mmproj_path.is_file():
            raise FileNotFoundError(
                f"mmproj 파일을 찾을 수 없습니다: {mmproj_path}\n"
                f"다운로드 방법:\n"
                f"  hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \\\n"
                f"      --include '*mmproj*' "
                f"--local-dir {settings.LLM_MODEL_PATH}"
            )
        cmd += ["--mmproj", str(mmproj_path),
                "--image-min-tokens", "1024",        # Qwen-VL grounding 정확도 권장 값
                "--image-max-tokens", str(settings.FRAME_EXTRACT_RESOLUTION)]
        
    mode_label = "멀티모달(VLM)" if multimodal else "텍스트 전용"
    logger.info(
        f"llama-server 시작 | {mode_label} | "
        f"모델: {model_path.name} | 포트: {settings.LLM_SERVER_PORT}"
    )

    # 서브프로세스 실행 - stdout/stderr PIPE로 캡쳐 (터미널 로그 오염 방지)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # /health 엔드포인트로 서버 준비 상태 확인 (모델 로딩 완료까지 대기)
    try:
        _wait_for_health(settings.LLM_SERVER_PORT, settings.LLM_SERVER_TIMEOUT)
    except TimeoutError:
        # 타임아웃 시 프로세스 정리 후 예외 전파
        proc.kill()
        proc.wait(timeout=5)
        raise

    logger.info(f"llama-server 준비 완료 | PID: {proc.pid} | {mode_label}")
    return proc

# --------------------------------------------------------------
# 서버 종료
# --------------------------------------------------------------

def stop_llm_server(proc: Optional[subprocess.Popen]) -> None:
    """
    llama-server 서브프로세스를 안전하게 종료

    프로세스 종료 시 해당 프로세스가 점유한 VRAM이 OS 수준에서 완전 해제됨
    gc.collect() + torch.cuda.empty_cache()는 불필요 (별도 프로세스이므로)

    종료 순서:
        1. SIGTERM 전송 (정상 종료 요청)
        2. 10초 대기
        3. 타임아웃 시 SIGKILL 강제 종료

    Args:
        proc: start_llm_server()가 반환한 Popen 객체 (None이면 무시)
    """

    if proc is None:
        return
    
    pid = proc.pid
    logger.info(f"llama-server 종료 시작 | PID: {pid}")

    proc.terminate()            # SIGTERM 전송
    try:
        proc.wait(timeout=10)   # 최대 10초 대기
    except subprocess.TimeoutExpired:
        logger.warning(f"llama-server SIGTERM 타임아웃, SIGKILL 전송 | PID: {pid}")
        proc.kill()             # 강제 종료
        proc.wait(timeout=5)

    logger.info(f"llama-server 종료 완료 |  PID: {pid} | VRAM 자동 해제됨")

# --------------------------------------------------------------
# 헬스 체크 (내부용)
# --------------------------------------------------------------

def _wait_for_health(port: int, timeout: int = 60) -> None:
    """
    llama-server의 /health 엔드포인트를 폴링하여 준비 상태 확인

    llama-server는 모델 로딩이 끝나야 요청을 처리할 수 있으므로,
    health check가 HTTP 200을 반환할 때까지 1초 간격으로 폴링

    표준 라이브러리 urllib만 사용 (httpx/requests 의존성 추가 불필요)

    Args:
        port: llama-server 포트 번호
        timeout: 최대 대기 시간 (초, 기본 60초)
    
    Raises:
        TimeoutError: 타임아웃 내에 서버가 준비되지 않음
    """

    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    attempt = 0

    while time.time() - start < timeout:
        attempt += 1
        try:
            resp = urlopen(url, timeout=2)
            if resp.status == 200:
                elapsed = round(time.time() - start, 1)
                logger.info(
                    f"llama-server health OK | "
                    f"{elapsed}초 소요 | {attempt}회 시도"
                )
                return
        except (URLError, OSError):
            pass                # 아직 서버 미기동 - 재시도

        time.sleep(1)           # 1초 간격 폴링

    raise TimeoutError(
        f"llama-server가 {timeout}초 내에 준비되지 않았습니다 "
        f"(포트: {port}, 시도: {attempt}회)"
    )