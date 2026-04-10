# 계층: API 계층 (Controller)
# 역할: 시스템 상태 확인 엔드포인트, GPU 정보 조회
# 의존: 없음 (외부 프로세스 nvidia-smi만 호출)
# MVA 원칙: 시스템 모니터링 기반 구축 (품질 속성: 관측 가능성)

"""
시스템 상태 엔드포인트
"""

# nvidia-smi 같은 외부 명령 실행
import subprocess

from fastapi import APIRouter

from app.schemas.api import GPUStatus, SysterStatus

router = APIRouter()

@router.get("/status", response_model=SysterStatus)
async def system_status():
    """
    시스템 상태 확인

    GET /api/v1/system/status
    Response: {
        "status": "healthy",
        "gpu": {"available": true, "name": "RTX 5070 Ti", "vram_total_mb": 12188, ...}
        "models_loaded": []
    }

    용도:
        - 서버 헬스 체크 (로드밸런서, 모니터링 도구)
        - GPU VRAM 사용량 모니터링 (모델 로드 전 여유 VRAM 확인)
        - 현재 GPU에 로드된 모델 목록 확인 (2주차 구현)
    """
    
    gpu = _check_gpu()
    return SysterStatus(
        status="healthy",
        gpu=gpu,
        models_loaded=[],        # 2주차 모델 매니저 구현 시 실제 목록 반환
    )

def _check_gpu() -> GPUStatus:
    """
    nvidia-smi를 톨한 GPU 상태 조회

    nvidia-smi: NVIDIA GPU 모니터링 CLI 도구
    --query-gpu: 조회할 GPU 속성 지정
    --format=csg,noheader,nounits: 파싱하기 쉬운 CSV 형식, 헤더/단위 제거

    출력 예시: "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12188, 543"
        -> 이름, 전체 VRAM(MB), 사용 중 VRAM(MB)

    GPU가 없거나 nvidia-smi가 설치되지 않은 환경에서는 
    available=False를 반환하여 정상 동작한다 (에러 전파 안함)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,        # stdout, strerr를 캡쳐
            text=True,                  # 바이트가 아닌 문자열로 반환
            timeout=5,                  # 5초 내 응답이 없으면 타임아웃
        )
        if result.returncode == 0 and result.stdout.strip():
            # CSV 파싱: "RTX 5070 Ti, 12188, 543" -> ["RTX 5070 Ti", "12188", "543"]
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return GPUStatus(
                available=True,
                name=parts[0],                          # GPU 이름
                vram_total_mb=int(float(parts[1])),     # 전체 VRAM (MB)
                vram_used_mb=int(float(parts[2])),      # 사용 중 VRAM (MB)
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # FileNotFoundError: nvidia-smi 가 설치되지 않음 (GPU가 없는 환경)
        # TimeoutExpired: GPU 드라이버 응답 없음
        pass

    return GPUStatus(available=False)                   # GPU 사용 불가