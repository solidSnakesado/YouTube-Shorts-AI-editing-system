"""
기본 API 테스트
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

@pytest_asyncio.fixture
async def client():
    from app.core.database import init_db
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_create_project(client):
    response = await client.post(
        "/api/v1/projects/",
        json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert "id" in data