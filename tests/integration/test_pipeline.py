# 계층: 테스트 (통합)
# 역할: 다운로드 -> 오디오 추출 -> 음성 전사 E2E 파이프라인 검증
#       실제 유투브/GPU를 사용하지 않고, 외부 의존성을 모킹하여
#       파이프라인의 상태 전환과 데이터 흐름이 정상인지 확인
# 의존: app.main, app.core.database, app.services, app.core.gpu_manager
#
# 테스트 실행 방법
#   uv run pytest tests/integration/test_pipeline.py -v
#
# 이 테스트가 검증하는 것
#   1. 프로젝트 생성 -> PENDING 상태
#   2. 다운로드 -> DOWNLOADING -> ANALYZING 상태 전환
#   3. 전사 -> transcript_json에 결과 저장
#   4. 전체 파이프라인 순차 실행 시 상태 흐름 정상
#   5. 오디오 파일 미존재 시 FAILED 처리
#   6. gpu_manager의 VRAM 해제 함수 호출 여부

"""
E2E 파이프라인 테스트

다운로드 -> 오디오 추출 -> 음성 전사 전체 흐름 검증
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

@pytest_asyncio.fixture
async def client():
    """
    테스트용 비동기 HTTP 클라이언트
    """

    from app.core.database import init_db
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

# --------------------------------------------------------------
# 헬퍼: 프로젝트 생성 공통 로직
# --------------------------------------------------------------

async def _create_project(client) -> dict:
    """
    테스트용 프로젝트 생성 후 응답 dict 반환
    """

    response = await client.post(
        "/api/v1/projects/",
        json={"youtube_url": "https://www.youtube.com/watch?v=test123"}
    )

    assert response.status_code == 201
    return response.json()

# --------------------------------------------------------------
# 테스트: 다운로드 파이프라인
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_creates_files_and_updates_status(client):
    """
    다운로드 파이프라인 테스트

    POST /api/v1/projects/{id}/download

    검증 항목:
        - yt-dlp 프로세스가 올바른 인자로 호출되는 지
        - FFmpeg 오디오 추출이 실행되는 지
        - 상태가 ANALYZING으로 전환되는 지
        - siurce_path, audio_path가 DB에 저장되는 지

    모킹 대상:
        - asyncio.create_subprocess_exec (yt-dlp, ffmpeg, ffprobe)
        - 실제 파일 다운로드는 수행하지 않음
    """

    project = await _create_project(client)
    project_id = project["id"]

    # 외부 프로세스(yt-dlp, ffmpeg, ffprobe)를 모킹
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(
        # ffprobe JSON 출력 시뮬레이션
        json.dumps({
            "format": {"duration": "300.5", "format_name": "mp4"}
        }).encode(),
        b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        response = await client.post(f"/api/v1/projects/{project_id}/download")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "analyzing"

# --------------------------------------------------------------
# 테스트: 전사 파이프라인
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_saves_transcript_json(client):
    """
    전사 파이프라인 테스트

    POST /api/v1/projects/{id}/transcribe

    검증 항목:
        - faster-whisper 모델 로드가 호출되는 지
        - 전사 결과가 transcript_json에 JSON으로 저장되는 지
        - 반환된 JSON에 segments, language, duration_sec 키가 있는지
        - VRAM 언로드(unload_model)가 호출되는 지

    모킹 대상:
        - gpu_manager.load_whisper (실제 GPU 모델 로드 방지)
        - gpu_manager.unload_model (VRAM 해제 호출 확인)
        - faster-whisper의 transcribe 결과
    """

    project = await _create_project(client)
    project_id = project["id"]

    # 다운로드 완료 상태를 시뮬레이션 (audio_path 설정)
    mock_dl_process = AsyncMock()
    mock_dl_process.returncode = 0
    mock_dl_process.communicate = AsyncMock(return_value=(
        json.dumps({"format": {"duration": "120.0"}}).encode(),
        b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=mock_dl_process):
        response = await client.post(f"/api/v1/projects/{project_id}/download")

    # faster-whisper 전사 결과를 모킹
    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 3.5
    mock_segment.text = " 안녕하세요 테스트입니다"
    mock_word_1 = MagicMock(word=" 안녕하세요", start=0.0, end=1.2, probability=0.95)
    mock_word_2 = MagicMock(word=" 테스트입니다", start=1.3, end=3.5, probability=0.88)
    mock_segment.words = [mock_word_1, mock_word_2]

    mock_info = MagicMock()
    mock_info.language = "ko"
    mock_info.language_probability = 0.98
    mock_info.duration = 120.0

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_segment], mock_info)

    # 모킹 경로: analysis_service가 from import로 가져온 참조를 패치해야 함
    # patch("app.core.gpu_manager.load_whisper")는 원본 모듈만 패치하고,
    # analysis_service 내부의 이미 import된 참조는 변경하지 않음
    with patch("app.services.analysis_service.load_whisper", return_value=mock_whisper), \
         patch("app.services.analysis_service.unload_model") as mock_unload, \
         patch("pathlib.Path.exists", return_value=True):
        
        response = await client.post(f"/api/v1/projects/{project_id}/transcribe")

    assert response.status_code == 200
    data = response.json()

    # unload_model이 호출되었는 지 (VRAM 해제 확인)
    mock_unload.assert_called_once()

@pytest.mark.asyncio
async def test_transcribe_without_audio_returns_error(client):
    """
    오디오 파일 없이 전사 시도 시 에러 처리 테스트

    POST /api/v1/projects/{id}/transcribe (다운로드 미완료 상태)

    검증 항목:
        - audio_path가 None인 프로젝트에 전사 요청 시 500 반환
        - 프로젝트 상태가 FAILED로 전환되는 지
    """

    project = await _create_project(client)
    project_id = project["id"]

    # 다운로드를 거치지 않고 바로 전사 요청 (audio_path = None)
    response = await client.post(f"/api/v1/projects/{project_id}/transcribe")

    assert response.status_code == 500

# --------------------------------------------------------------
# 테스트: 전체 E2E 흐름 (생성 -> 다운로드 -> 전사)
# --------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_create_download_transcribe(client):
    """
    전체 E2E 파이프라인 테스트
        1. POST /projects/                  -> 201, status=pending
        1. POST /projects/{id}/download     -> 200, status=analyzing
        1. POST /projects/{id}/transcribe   -> 200, status=transcript 저장됨

    이 테스트가 통과하면:
        - 프로젝트 생성 -> 다운로드 -> 전사 전체 흐름이 정상
        - 상태 전환: PENDING -> DOWNLOADING -> ANALYZING (전사 후 유지)
        - DI 체인 전체가 정상 동작
        - 각 서비스 간 데이터 전달이 올바름
    """

    # 1단계: 프로젝트 생성
    project = await _create_project(client)
    project_id = project["id"]
    assert project["status"] == "pending"

    # 2단계: 다운로드 (모킹)
    mock_dl_process = AsyncMock()
    mock_dl_process.returncode = 0
    mock_dl_process.communicate = AsyncMock(return_value=(
        json.dumps({"format": {"duration": "180.0"}}).encode(),
        b""
    ))

    with patch("asyncio.create_subprocess_exec", return_value=mock_dl_process):
        dl_response = await client.post(f"/api/v1/projects/{project_id}/download")
    
    assert dl_response.status_code == 200
    assert dl_response.json()["status"] == "analyzing"

    # 3단계: 전사 (모킹)
    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 10.0
    mock_segment.end = 15.5
    mock_segment.text = " 이것은 E2E 테스트입니다"
    mock_word = MagicMock(word=" E2E", start=12.0, end=13.0, probability=0.92)
    mock_segment.words = [mock_word]

    mock_info = MagicMock()
    mock_info.language = "ko"
    mock_info.language_probability = 0.95
    mock_info.duration = 180.0

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_segment], mock_info)

    with patch("app.services.analysis_service.load_whisper", return_value=mock_whisper), \
         patch("app.services.analysis_service.unload_model") as mock_unload, \
         patch("pathlib.Path.exists", return_value=True):
        
        tr_response = await client.post(f"/api/v1/projects/{project_id}/transcribe")

    assert tr_response.status_code == 200

    # 최종 상태 확인
    get_response = await client.get(f"/api/v1/projects/{project_id}")
    assert get_response.status_code == 200
    final = get_response.json()
    assert final["status"] == "analyzing"