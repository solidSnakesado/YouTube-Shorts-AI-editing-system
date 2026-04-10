# 계층: 테스트
# 역할: API 엔드포인트의 기본 동작을 검증하는 단위 테스트
#       실제 서버를 가동하지 않고, ASGI Transpor로 FastAPI 앱에 직접 오청을 보냄
# 의존: app.main (FastAPI 앱), app.core.database (DB 초기화)
#
# 테스트 실행 방법:
#   uv run pytest tests/unit/test_api.py -v
#
# 주의 사항 (개발 중 발견된 이슈 및 해결)
#   1. pytest_asyncio.fixture 필수: pytest-asyncio 1.3+, 
#      strict 모드에서는 비동기 fixture에 @pytest.fixture가 아닌 pytest_asyncio.fixture를 사용해야 함
#   2. init_db() 호출 필수: 테스트시 FastAPI의 lifespan이 실행되지 않으므로 fixture에서 직접 DB 테이블을 생성해야 함
#   3. URL 끝 슬래시 주위: POST /api/v1/.projects  (슬래스 없음) -> 307 리다이렉트
#                        POST /api/v1/.projects/ (슬래스 있음) -> 201 정상 응답
"""
기본 API 테스트
"""

import pytest
import pytest_asyncio       # 비동기 fixture 전용 데코레이터

# ASGITransport: 실제 HTTP 서버 없이 ASGI 앱에 직접 요청을 보내는 트랜스포트
# AsyncClient: 비동기 HTTP 클라이언트 (requests 라이브러리의 비동기 버전)
from httpx import ASGITransport, AsyncClient

from app.main import app    # FastAPI 앱 인스턴스

@pytest_asyncio.fixture
async def client():
    """
    테스트용 비동기 HTTP 클라이언트 fixture

    각 테스트 함수 실행 전에 호출되어:
        1. DB 테이블 생성 (init_db)
        2. ASGI Transport 기반 클라이언트 생성
        3. 테스트에 클라이언트 제공
        4. 테스트 종료 후 클라이언트 자동 정리

    @pytest_asyncio.fixture 를 사용하는 이유:
        asyncio_mode = "strict" 설정에서 @pytest.fixture는 
        비동기 fixture를 인식하지 못해 PytestRemoveIn9Warning 발생

    init_db 를 호출 하는 이유:
        테스트 환경에서는 FastAPI의 lifespan 이벤트가 실행되지 않으므로,
        DB 테이블 (projects, shorts)이 존재하지 않는 상태
        fixture 에서 직접 초기화해야 INSERT 시 "no such table" 에러를 방지
    """
    
    from app.core.database import init_db
    await init_db()     # CREATE TABLE IF NOT EXISTS 실행

    # ASGITransport: 네트워크를 거치지 않고 FastAPI 앱에 직접 요청
    # 장점: 실제 서버 기종 불필요, 테스트 속도 빠름 (~0.3초)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac        # 테스트 함수에 클라이언트 전달

@pytest.mark.asyncio
async def test_health_check(client):
    """
    헬스 체크 엔드포인트 테스트

    GET /health -> 200 OK, {"status": "healthy"}

    이 테스트가 통과하면:
        - FastAPI 앱이 정상 기동됨
        - 미들웨어 등록 정상
        - 라우팅 정상
    """
    
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_create_project(client):
    """
    프로젝트 생성 엔드포인트 테스트

    POST /api/v1/projects/ -> 201 Created

    이 테스트가 통과하면
        - DI 체인 정상 (get_db -> get_project_repo -> get_video_service)
        - SQLite 테이블 생성 정상 (projects 테이블 존재)
        - Pydantic 스키마 검증 정상 (youtube_url 형식 확인)
        - 레포지토리 CRUD 정상 (BaseRepository.create 동작)
        - 응답 스키마 매핑 정상 (ProjectResponse)

    URL 끝에 슬래시(/) 필수:
    FastAPI의 라우터가 @router.post("/")로 정의되어 있으므로
    슬래시 없이 요청하면 307 Temporary Redirect가 반환
    """
    
    response = await client.post(
        "/api/v1/projects/",                # 슬래시 필수
        json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"      # 새 프로젝트는 항상 PENDING 상태
    assert "id" in data                     # UUID가 자동 생성되었는지 확인