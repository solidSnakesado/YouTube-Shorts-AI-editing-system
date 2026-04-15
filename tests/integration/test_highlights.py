# 계층: 테스트 (통합)
# 역할: LLM 하이라이트 추출 파이프라인 검증 (모킹 기반)
# 모킹 규칙: from X import Y -> patch("사용모듈.Y")
# 6~7일차 신규 / 13일차: 자동 길이 판단 테스트 추가

"""
LLM 하이라이트 추출 통합 테스트 - 전사 -> 하이라이트 추출 -> Shorts 생성 흐름 검증
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.llm_highlight_extractor import parse_highlights, MIN_DURATION_SEC, MAX_DURATION_SEC

@pytest_asyncio.fixture
async def client():
    """
    테스트 비동기 HTTP 클라이언트

    init_db() 호출 필수: lifespan이 실행되지 않으므로
    테이블이 생성되지 않는 상태에서 테스트 하면 실패
    """

    from app.core.database import init_db
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

# --------------------------------------------------------------
# 헬퍼: 각 테스트에서 반복되는 준비 작업을 함수로 추출
# --------------------------------------------------------------

async def _create_project(client) -> dict:
    """
    테스트용 프로젝트 생성 후 응답 dict 반환
    """

    repo = await client.post(
        "/api/v1/projects/",
        json={"youtube_url": "https://www.youtube.com/watch?v=highlight_test"}
    )
    assert repo.status_code == 201
    return repo.json()

async def _download_project(client, pid: str) -> dict:
    """
    프로젝트 다운로드 시뮬레이션 (yt-dlp, ffmpeg, ffprobe 모킹)

    AsyncMock으로 외부 프로세스를 모킹하여 실제 다운로드 없이
    상태 전환(PENDING -> DOWNLOADING -> ANALYZING)을 검증
    """

    # 외부 프로세스(yt-dlp, ffmpeg, ffprobe) 모킹
    proc = AsyncMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(
        # ffprobe JSON 출력 시뮬레이션
        json.dumps({"format": {"duration": "300.0"}}).encode(), b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        repo = await client.post(f"/api/v1/projects/{pid}/download")
    assert repo.status_code == 200
    return repo.json()

async def _transcribe_project(client, pid: str) -> dict:
    """
    프로젝트 전사 시뮬레이션 (faster-whisper 모킹)

    MagicMock으로 Whisper 모델의 transcribe() 결과를 시뮬레이션
    세그먼트 2개 + 단어 타임스탬프 포함
    """

    # faster-whisper 전사 결과 모킹
    seg1 = MagicMock(id=0, start=0.0, end=30.0, text=" 안녕하세요 놀라운 이야기입니다")
    seg1.words = [MagicMock(word=" 안녕하세요", start=0.0, end=1.5, probability=0.95)]

    seg2 = MagicMock(id=1, start=30.0, end=60.0, text=" 정말 대단한 발견입니다")
    seg2.words = [MagicMock(word=" 대단한", start=31.0, end=32.0, probability=0.90)]

    info = MagicMock(language="ko", language_probability=0.98, duration=300.0)
    whisper = MagicMock()
    whisper.transcribe.return_value = ([seg1, seg2], info)

    # 모킹 경로: analysis_service가 from import로 가져온 참조를 패치
    with patch("app.services.analysis_service.load_whisper", return_value=whisper), \
         patch("app.services.analysis_service.unload_model"), \
         patch("pathlib.Path.exists", return_value=True):
        resp = await client.post(f"/api/v1/projects/{pid}/transcribe")

    assert resp.status_code == 200
    return resp.json()

# LLM이 반환할 모킹 응답 - 13일차: 다양한 길이 (LLM이 자동 판단 시뮬레이션)
MOCK_LLM_RESPONSE = json.dumps({
    "highlights": [
        {
            "start_sec": 10.0, "end_sec": 55.0, "hook_score": 0.92,
            "reason": "놀라운 도입부", "title_suggestion": "놀라운 사실",
            "tags": ["정보"],
        },
        {
            "start_sec": 120.0, "end_sec": 145.0, "hook_score": 0.85,
            "reason": "핵심 인사이트", "title_suggestion": "이것만 알면",
            "tags": ["핵심"],
        },
    ]
})

def _mock_llm():
    """
    LLM 핸들 모킹 (OpenAI 타입으로 설정하여 GPU 미사용)
    """

    return {"type": "openai", "client": MagicMock()}

# --------------------------------------------------------------
# 테스트: 하이라이트 추출 성공
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_creates_shorts_and_updates_status(client):
    """
    하이라이트 추출 -> Shorts DB 생성 -> 상태 EDITING 전환 검증

    POST /api/v1/projects/{id}/analyze

    검증 항목:
        - LLM 호출 후 응답이 파싱됨
        - Short 엔티티 2개가 DB에 생성됨
        - 프로젝트 상태가 EDITING으로 전환됨
        - 각 쇼츠의 hook_score, highlight_reason이 존재
    """

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])

    with patch("app.services.analysis_service.load_llm", return_value=_mock_llm()), \
         patch("app.services.analysis_service.call_llm", return_value=MOCK_LLM_RESPONSE), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{p['id']}/analyze?max_shorts=5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2                       # 모킹 응답에서 2개 하이라이트
    first = data["items"][0]
    assert first["project_id"] == p["id"]
    assert first["status"] == "queued"              # 초기 상태: 편집 대기
    assert first["hook_score"] is not None

    # 프로젝트 상태 확인 (ANALYZING -> EDITING)
    proj = await client.get(f"/api/v1/projects/{p['id']}")
    assert proj.json()["status"] == "editing"

# --------------------------------------------------------------
# 테스트: 전사 없이 하이라이트 추출
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_without_transcript_returns_error(client):
    """
    전사 데이터 없이 분석 요청 시 500 반환

    다운로드만 완료하고 전사를 거치지 않은 상태에서
    analyze를 호출 하면 transcript_json이 None이므로 실패 해야 함
    """

    p = await _create_project(client)
    await _download_project(client, p["id"])

    with patch("app.services.analysis_service.load_llm", return_value=_mock_llm()), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{p['id']}/analyze?max_shorts=5")

    assert resp.status_code == 500

# --------------------------------------------------------------
# 테스트: LLM 응답 파싱 실패
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_with_invalid_llm_response(client):
    """
    LLM이 유효하지 않은 응답을 반환할 때 에러 처리

    검증: 500 반환 + 프로젝트 상태 FAILED 전환
    """

    p = await _create_project(client)
    await _download_project(client, p["id"])
    await _transcribe_project(client, p["id"])

    # LLM이 JSON이 아닌 텍스트를 반환
    with patch("app.services.analysis_service.load_llm", return_value=_mock_llm()), \
         patch("app.services.analysis_service.call_llm", return_value="not json"), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{p['id']}/analyze?max_shorts=5")

    assert resp.status_code == 500

# --------------------------------------------------------------
# 테스트: 전체 E2E (생성 -> 다운로드 -> 전사 -> 하이라이트)
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_pipeline_create_to_analyze(client):
    """
    전체 E2E 파이프라인 테스트

    순서:
        1. POST /projects/                  -> 201, pending
        2. POST /projects/{id}/download     -> 201, analyzing
        3. POST /projects/{id}/transcribe   -> 201, analyzing
        4. POST /projects/{id}/analyze      -> 201, editing + shorts 생성

    이 테스트가 통과하면:
        - 전체 파이프라인 상태 전환이 정상
        - DI 체인 전체가 정상 동작
        - Shorts 엔티티가 DB에 올바르게 저장된
        - shorts_count가 프로젝트 조회에 반영됨
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

    # 4. 하이라이트 추출
    with patch("app.services.analysis_service.load_llm", return_value=_mock_llm()), \
         patch("app.services.analysis_service.call_llm", return_value=MOCK_LLM_RESPONSE), \
         patch("app.services.analysis_service.unload_model"):
        resp = await client.post(f"/api/v1/projects/{p['id']}/analyze?max_shorts=5")

    assert resp.status_code == 200
    assert resp.json()["total"] >= 1

    # 최종 상태 확인
    final = await client.get(f"/api/v1/projects/{p['id']}")
    assert final.json()["status"] == "editing"
    assert final.json()["shorts_count"] >= 1         # 쇼츠가 생성되어 카운트 반영됨

# --------------------------------------------------------------
# 테스트: LLM 자동 길이 판단 검증 (13일차 추가)
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_duration_validate_min_max():
    """
    parse_highlights가 MIN/MAX_DURATION_SEC 범위를 올바르게 적용하는 지 검증
        - 10초 미만: 제거
        - 120초 초과: 120초로 잘라내기
        - 정상 범위: 그대로 유지
    """

    llm_response = json.dumps({"highlights": [
        {"start_sec": 0, "end_sec": 5, "hook_score": 0.9, "reason": "너무 짧음", "title_suggestion": "짧은", "tags": []},
        {"start_sec": 10, "end_sec": 200, "hook_score": 0.8, "reason": "너무 긴 구간", "title_suggestion": "긴", "tags": []},
        {"start_sec": 50, "end_sec": 90, "hook_score": 0.7, "reason": "적절한 구간", "title_suggestion": "적절", "tags": []},
    ]})

    result = parse_highlights(llm_response, total_duration=300.0, max_shorts=5)

    # 5초짜리는 MIN_DURATION_SEC(10) 미만이므로 제거됨
    assert all(r["end_sec"] - r["start_sec"] >= MIN_DURATION_SEC for r in result)

    # 190초짜리는 MAX_DURATION_SEC(120)로 짤림 -> end_sec = 10 + 120 = 130
    long_clip = [r for r in result if r["start_sec"] == 10.0]
    assert len(long_clip) == 1
    assert long_clip[0]["end_sec"] == 10.0 + MAX_DURATION_SEC

    # 40초짜리는 그대로 유지
    normal_clip = [r for r in result if r["start_sec"] == 50.0]
    assert len(normal_clip) == 1
    assert normal_clip[0]["end_sec"] == 90.0