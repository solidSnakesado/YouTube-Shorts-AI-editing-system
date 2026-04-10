"""
시스템 상태 엔드포인트
"""

import subprocess

from fastapi import APIRouter

from app.schemas.api import GPUStatus, SysterStatus

router = APIRouter()

@router.get("/status", response_model=SysterStatus)
async def system_status():
    gpu = _check_gpu()
    return SysterStatus(
        status="healthy",
        gpu=gpu,
        model_loaded=[],
    )

def _check_gpu() -> GPUStatus:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return GPUStatus(
                available=True,
                name=parts[0],
                vram_total_mb=int(float(parts[1])),
                vram_used_mb=int(float(parts[2])),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return GPUStatus(available=False)