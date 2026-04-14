# 계층: 테스트 (통합)
# 역할: 자막 생성 + 최종 인코딩 파이프라인 검증
#       실제 FFmpeg/GPU를 사용하지 않고, 외부 의존성을 모킹하여
#       자막 생성 -> 인코딩 -> COMPLETED 상태 전환을 검증
# 의존: app.main, app.core.database, app.services
#
# 테스트 실행 방법
#   uv run pytest tests/integration/test_encoding.py -v
#
# 이 테스트가 검증하는 것
#   1. 자막 생성 성공 -> output_path 업데이트
#   2. 리프레이밍 없이 자막 생성 시 500 에러
#   3. 최종 인코딩 성공 -> status=completed + output_path 저장
#   4. 전체 E2E: 생성 -> 다운로드 -> 전사 -> 분석 -> 편집 -> 자막 -> 인코딩 -> 완료
#
# 모킹 경로 규칙:
#   patch("app.services.editing_service.run_ffmpeg_subtitle")       - subtitle_genetator 헬퍼
#   patch("app.services.editing_service.run_ffmpeg_encode")         - subtitle_genetator 헬퍼
#   patch("app.services.editing_service.extract_words_for_range")   - subtitle_genetator 헬퍼
#   patch("app.services.editing_service.write_ass_file")            - subtitle_genetator 헬퍼
#
# 11~12일차 신규 파일

"""
자막 생성 + 최종 인코딩 통합 테스트
리프레이밍 -> 자막 -> 인코딩 -> COMPLETED 흐름 검증
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
# 헬퍼: 이전 단계 시뮬레이션 (test_reframe.py 패턴 재사용)
# --------------------------------------------------------------

async def _create_project(client) -> dict:
    """프로젝트 생성"""

    resp = await client.post(
        "/api/v1/projects/",
        json={"youtube_url": "https://www.youtube.com/watch?v=encoding_test"}
    )

    assert resp.status_code == 201
    return resp.json()

async def _download_project(client, pid: str) -> dict:
    """다운로드 시뮬레이션"""

    proc = AsyncMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(
        json.dumps({"format": {"duration": "300.0"}}).encode(), b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        resp = await client.post(f"/api/v1/projects/{pid}/download")
    
    assert resp.status_code == 200
    return resp.json()

async def _transcribe_project(client, pid: str) -> dict:
    """전사 시뮬레이션 (Whisper 모킹, 단어 타임스탬프 포함)"""

    seg1 = MagicMock(id=0, start=0.0, end=30.0, text=" 안녕하세요 놀라운 이야기입니다")
    seg1.words = [
        MagicMock(word=" 안녕하세요", start=0.0, end=1.5, probability=0.95),
        MagicMock(word=" 놀라운", start=1.6, end=2.5, probability=0.90),
        MagicMock(word=" 이야기입니다", start=2.6, end=4.0, probability=0.88),
    ]
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

# LLM 모킹 응답
MOCK_LLM_RESPONSE = json.dumps({
    "highlights": [{
        "start_sec": 0.0, "end_sec": 60.0, "hook_score": 0.92,
        "reason": "놀라운 도입부", "title_suggestion": "놀라운 사실", "tags": ["정보"],
    }]
})

async def _analyze_project(client, pid: str) -> dict:
    """하이라이트 추출 시뮬레이션"""

    mock_llm = {"type": "openai", "client": MagicMock()}
    with patch("app.services.analysis_service.load_llm", return_value=mock_llm), \
         patch("app.services.analysis_service.call_llm", return_value=MOCK_LLM_RESPONSE), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{pid}/analyze?max_shorts=5") 

    assert resp.status_code == 200
    return resp.json()

def _mock_yolo():
    """YOLO 모델 모킹 (3프레임 탐지 결과)"""
    
    import numpy as np
    frames = []
    for i in range(3):
        result = MagicMock()
        result.orig_shape = (1080, 1920)
        box = MagicMock()
        xywh_data = np.array([[960 + i * 20, 540, 200, 400]], dtype=np.float32)
        box_xywh = MagicMock()
        box_xywh.__getitem__ = lambda self, key, d=xywh_data: MagicMock(
            __mul__=lambda s, other: MagicMock(
                cpu=lambda: MagicMock(numpy=lambda: np.array([80000.0]))))
        box.xywh.cpu = lambda d=xywh_data: MagicMock(numpy=lambda: d[0])
        result.boxes = box
        frames.append(result)
    
    model = MagicMock()
    model.predict.return_value = iter(frames)
    return model

async def _reframe_short(client, shorts_id: str) -> dict:
    """리프레이밍 시뮬레이션"""

    ffmpeg_ok = AsyncMock(returncode=0)
    ffmpeg_ok.communicate = AsyncMock(return_value=(b"", b""))
    with patch("app.services.editing_service.load_yolo", return_value=_mock_yolo()), \
         patch("app.services.editing_service.unload_model"), \
         patch("asyncio.create_subprocess_exec", return_value=ffmpeg_ok), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/edit") 

    assert resp.status_code == 200
    return resp.json()

# --------------------------------------------------------------
# 테스트 1: 자막 생성 성공
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_subtitle_generates_and_updates_output(client):
    """자막 생성 -> output_path 업데이트 검증"""

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze = await _analyze_project(client, p["id"])
    shorts_id = analyze["items"][0]["id"]
    await _reframe_short(client, shorts_id)

    # 자막 생성 모킹: ASS 파일 작성 + FFmpeg 자막 합성
    ffmpeg_ok = AsyncMock(returncode=0)
    ffmpeg_ok.communicate = AsyncMock(return_value=(b"", b""))

    with patch("app.services.editing_service.extract_words_for_range", return_value=[
            {"word": "안녕", "start": 0.0, "end": 0.5},
            {"word": "하세요", "start": 0.6, "end": 1.0},
         ]), \
         patch("app.services.editing_service.write_ass_file", return_value=True), \
         patch("asyncio.create_subprocess_exec", return_value=ffmpeg_ok), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/subtitle") 

    assert resp.status_code == 200
    data = resp.json()
    assert data["output_path"] is not None
    assert "subtitled" in data["output_path"]

# --------------------------------------------------------------
# 테스트 2: 리프레이밍 없이 자막 생성 시도
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_subtitle_without_reframe_returns_error(client):
    """리프레이밍 미완료 상태에서 자막 생성 시 500 반환"""
    
    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze = await _analyze_project(client, p["id"])
    shorts_id = analyze["items"][0]["id"]

    # 리프레이밍을 거치지 않고 바로 자막 생성 (output_path = None)
    with patch("pathlib.Path.exists", return_value=False):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/subtitle") 

    assert resp.status_code == 500

# --------------------------------------------------------------
# 테스트 3: 최종 인코딩 성공
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_encode_final_completes_short(client):
    """최종 인코딩 -> status=completed + output_path 저장"""

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])
    analyze = await _analyze_project(client, p["id"])
    shorts_id = analyze["items"][0]["id"]
    await _reframe_short(client, shorts_id)

    # 자막 먼저 실행
    ffmpeg_ok = AsyncMock(returncode=0)
    ffmpeg_ok.communicate = AsyncMock(return_value=(b"", b""))
    with patch("app.services.editing_service.extract_words_for_range", return_value=[
            {"word": "테스트", "start": 0.0, "end": 0.5}
         ]), \
         patch("app.services.editing_service.write_ass_file", return_value=True), \
         patch("asyncio.create_subprocess_exec", return_value=ffmpeg_ok), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/subtitle") 

    # 최종 인코딩 실행
    ffmpeg_enc = AsyncMock(returncode=0)
    ffmpeg_enc.communicate = AsyncMock(return_value=(b"", b""))
    with patch("asyncio.create_subprocess_exec", return_value=ffmpeg_enc), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/encode") 

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["output_path"] is not None

# --------------------------------------------------------------
# 테스트 4: 전체 E2E (생성 -> 다운로드 -> 전사 -> 분석 -> 편집 -> 자막 -> 인코딩)
# --------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_pipeline_create_to_completed(client):
    """전체 E2E: 7단계 파이프라인 -> status=completed"""

    # 1. 생성
    p = await _create_project(client)
    assert p["status"] == "pending"

    # 2. 다운로드
    dl = await _download_project(client, p["id"])
    assert dl["status"] == "analyzing"

    # 3. 전사
    tr = await _transcribe_project(client, p["id"])
    assert tr["status"] == "analyzing"

    # 4. 하이라이트 추출
    analyze = await _analyze_project(client, p["id"])
    assert analyze["total"] >= 1
    shorts_id = analyze["items"][0]["id"]

    # 5. 리프레이밍
    rf = await _reframe_short(client, shorts_id)
    assert rf["output_path"] is not None

    # 6. 자막 생성
    ffmpeg_ok = AsyncMock(returncode=0)
    ffmpeg_ok.communicate = AsyncMock(return_value=(b"", b""))
    with patch("app.services.editing_service.extract_words_for_range", return_value=[
            {"word": "E2E", "start": 0.0, "end": 0.5}
         ]), \
         patch("app.services.editing_service.write_ass_file", return_value=True), \
         patch("asyncio.create_subprocess_exec", return_value=ffmpeg_ok), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/shorts/{shorts_id}/subtitle") 

    assert resp.status_code == 200

    # 최종 인코딩 실행
    ffmpeg_enc = AsyncMock(returncode=0)
    ffmpeg_enc.communicate = AsyncMock(return_value=(b"", b""))
    with patch("asyncio.create_subprocess_exec", return_value=ffmpeg_enc), \
         patch("pathlib.Path.exists", return_value=True):
        enc = await client.post(f"/api/v1/shorts/{shorts_id}/encode") 

    assert enc.status_code == 200
    assert enc.json()["status"] == "completed"
    
    # 프로젝트 상태는 editing 유지 (모든 쇼츠 완료 시 COMPLETED 전환은 13~14일차)
    proj = await client.get(f"/api/v1/projects/{p['id']}")
    assert proj.json()["status"] == "editing"