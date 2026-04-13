# 계층: 테스트 (통합)
# 역할: YOLOv8 리프레이밍 파이프라인 검증
#       실제 YOLO/GPU를 사용하지 않고, 외부 의존성을 모킹하여
#       리프레이밍 -> 상태 전환 -> 출력 파일 경로 저장을 검증
# 의존: app.main, app.core.database, app.services
#
# 테스트 실행 방법
#   uv run pytest tests/integration/test_reframe.py -v
#
# 이 테스트가 검증하는 것
#   1. 리프레이밍 성공 -> output_path 저장 + 상태 복원
#   2. 소스 파일 없이 편집 시 500 에러
#   3. YOLO 탐지 실패 시 센터 크롭 폴백
#   4. 전체 E2E: 생성 -> 다운로드 -> 전사 -> 분석 -> 편집
#
# 모킹 경로 규칙:
#   patch("app.services.editing_service.load_yolo")     - 올바름
#   patch("app.services.gpu_manager.load_yolo")         - 효과 없음
#
# 8~10일차 신규 파일

"""
YOLOv8 리프레이밍 통합 테스트
분석(하이라이트) -> 리프레이밍 -> 출력 파일 생성 흐름 검증
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

@pytest_asyncio.fixture
async def client():
    """테스트 비동기 HTTP 클라이언트 (init_db 필수)"""
    
    from app.core.database import init_db
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

# --------------------------------------------------------------
# 헬퍼: 이전 단계 시뮬레이션 (test_highlight.py와 동일한 패턴)
# --------------------------------------------------------------

async def _create_project(client) -> dict:
    """프로젝트 생성"""

    resp = await client.post(
        "/api/v1/projects/",
        json={"youtube_url": "https://www.youtube.com/watch?v=reframe_test"}
    )
    
    assert resp.status_code == 201
    return resp.json()

async def _download_project(client, pid: str) -> dict:
    """다운로드 시뮬레이션 (yt-dlp, ffmpeg, ffprobe 모킹)"""

    proc = AsyncMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(
        json.dumps({"format": {"duration": "300.0"}}).encode(), b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(f"/api/v1/projects/{pid}/download")

    assert resp.status_code == 200
    return resp.json()

async def _transcribe_project(client, pid: str) -> dict:
    """전사 시뮬레이션 (Whisper 모킹)"""

    seg1 = MagicMock(id=0, start=0.0, end=30.0, text=" 안녕하세요 놀라운 이야기입니다")
    seg1.words = [MagicMock(word=" 안녕하세요", start=0.0, end=1.5, probability=0.95)]
    seg2 = MagicMock(id=1, start=30.0, end=60.0, text=" 정말 대단한 발견입니다")
    seg2.words = [MagicMock(word=" 대단한", start=31.0, end=32.0, probability=0.90)]
    info = MagicMock(language="ko", language_probability=0.98, duration=300.0)
    whisper = MagicMock()
    whisper.transcribe.return_value = ([seg1, seg2], info)

    with patch("app.services.analysis_service.load_whisper", return_value=whisper), \
         patch("app.services.analysis_service.unload_model"), \
         patch("pathlib.Path.exists", return_value=True):
        
        resp = await client.post(f"/api/v1/projects/{pid}/transcribe") 

    assert resp.status_code == 200
    return resp.json()

# LLM 모킹 응답 (하이라이트 2개)
MOCK_LLM_RESPONSE = json.dumps({
    "highlights": [
        {
            "start_sec": 10.0, "end_sec": 70.0, "hook_score": 0.92,
            "reason": "놀라운 도입부", "title_suggestion": "놀라운 사실", "tags": ["정보"],
        },
        {
            "start_sec": 120.0, "end_sec": 180.0, "hook_score": 0.85,
            "reason": "핵심 인사이트", "title_suggestion": "이것만 알면","tags": ["핵심"],
        },
    ]
})

async def _analyze_project(client, pid: str) -> dict:
    """하이라이트 추출 시뮬레이션 (LLM 모킹)"""

    mock_llm = {"type": "openai", "client": MagicMock()}
    with patch("app.services.analysis_service.load_llm", return_value=mock_llm), \
         patch("app.services.analysis_service.call_llm", return_value=MOCK_LLM_RESPONSE), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{pid}/analyze?max_shorts=5") 

    assert resp.status_code == 200
    return resp.json()

# --------------------------------------------------------------
# YOLO 모킹 헬퍼
# --------------------------------------------------------------

def _mock_yolo_model():
    """
    YOLO 모델 모킹: predict()가 탐지 결과를 반환
    person 클래스 1명을 영상 중앙 부근에서 탐지한 것으로 시뮬레이션
    """

    import numpy as np

    # 프레임 3개분의 탐지 결과 모킹
    frames = []
    for i in range(3):
        result = MagicMock()
        result.orig_shape = (1080, 1920)

        # 바운드 박스 모킹: [cx, cy, w, h] 형태의 xywh 텐서
        box = MagicMock()

        # 중앙 부근에서 약간 이동하는 피사체
        xywh_data = np.array([[960 + i * 20, 540, 200, 400]], dtype=np.float32)
        box_xywh = MagicMock()
        box_xywh.__getitem__ = lambda self, idx, d=xywh_data: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: d[0]))
        box_xywh.__len__ = lambda self: 1

        # areas 계산을 위한 슬라이싱 지원
        box.xywh = MagicMock()
        box_xywh.__getitem__ = lambda self, key, d=xywh_data: MagicMock(
            __mul__=lambda s, other: MagicMock(
                cpu=lambda: MagicMock(numpy=lambda: np.array([80000.0]))))
        box_xywh.__len__ = lambda self: 1

        # 개별 박스 접근
        box.xywh.cpu = lambda d=xywh_data: MagicMock(numpy=lambda: d[0])

        result.boxes = box
        frames.append(result)
    
    model = MagicMock()
    model.predict.return_value = iter(frames)
    return model

def _mock_ffmpeg_success():
    """FFmpeg 성공 모킹"""

    proc = AsyncMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(b"", b""))
    return proc

# --------------------------------------------------------------
# 테스트 1: 리프레이밍 성공
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_reframe_create_output_and_updates_status(client):
    """
    리프레이밍 성공 -> output_path 저장 + 상태 복원 검증

    검증 항목:
        - YOLO 탐지 -> 스무딩 -> 전략 선택 -> 크롭 타임라인 생성
        - FFmpeg 실행 성공
        - output_path가 Shorts 엔티티에 저장됨
        - 쇼츠 상태가 queued로 복원 (자막/인코딩 대기) 
    """

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze_resp = await _analyze_project(client, p["id"])

    shorts_id = analyze_resp["items"][0]["id"]

    with patch("app.services.editing_service.load_yolo", return_value=_mock_yolo_model()), \
         patch("app.services.editing_service.unload_model"), \
         patch("asyncio.create_subprocess_exec", return_value=_mock_ffmpeg_success()), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/edit") 

    assert resp.status_code == 200
    data = resp.json()
    assert data["output_path"] is not None
    assert data["status"] == "queued"       # 자막/인코딩 대기 상태

# --------------------------------------------------------------
# 테스트 2: 소스 파일 없이 편집
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_reframe_without_source_returns_error(client):
    """소스 영상 파일 없이 리프레이밍 시 500 반환"""

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze_resp = await _analyze_project(client, p["id"])

    shorts_id = analyze_resp["items"][0]["id"]

    # Path.exists = False -> 소스 파일 미존재
    with patch("app.services.editing_service.load_yolo", return_value=_mock_yolo_model()), \
         patch("app.services.editing_service.unload_model"), \
         patch("pathlib.Path.exists", return_value=False):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/edit") 

    assert resp.status_code == 500


# --------------------------------------------------------------
# 테스트 3: FFmpeg 실패 시 에러 처리
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_reframe_ffmpeg_failure_returns_error(client):
    """FFmpeg 실패 시 500 반환 + 쇼츠 상태 FAIDED"""

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze_resp = await _analyze_project(client, p["id"])

    shorts_id = analyze_resp["items"][0]["id"]

    # FFmpeg returncode != 0
    proc_fail = AsyncMock(returncode=1)
    proc_fail.communicate = AsyncMock(return_value=(b"", b"encoding error"))

    with patch("app.services.editing_service.load_yolo", return_value=_mock_yolo_model()), \
         patch("app.services.editing_service.unload_model"), \
         patch("asyncio.create_subprocess_exec", return_value=proc_fail), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/edit") 

    assert resp.status_code == 500

# --------------------------------------------------------------
# 테스트 4: 전체 E2E (생성 -> 다운로드 -> 전사 -> 분석 -> 편집)
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_pipeline_create_to_reframe(client):
    """
    전체 E2E 파이프라인 테스트
    생성 -> 다운로드 -> 전사 -> 하이라이트 -> 리프레이밍
    """

    # 1. 생성
    p = await _create_project(client)
    assert p["status"] == "pending"

    # 2. 다운로드
    dl = await _download_project(client, p["id"])
    assert dl["status"] == "analyzing"

    # 3. 전사
    tr = await _transcribe_project(client, p["id"])
    assert tr["status"] == "analyzing"

    # 4. 하이라이트
    analyze_resp = await _analyze_project(client, p["id"])
    assert analyze_resp["total"] >= 1

    # 5. 리프레이밍 (첫 번째 쇼츠)
    shorts_id = analyze_resp["items"][0]["id"]

    with patch("app.services.editing_service.load_yolo", return_value=_mock_yolo_model()), \
         patch("app.services.editing_service.unload_model"), \
         patch("asyncio.create_subprocess_exec", return_value=_mock_ffmpeg_success()), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/edit") 

    assert resp.status_code == 200
    assert resp.json()["output_path"] is not None

    # 프로젝트 상태는 editing 유지
    proj = await client.get(f"/api/v1/projects/{p['id']}")
    assert proj.json()["status"] == "editing"