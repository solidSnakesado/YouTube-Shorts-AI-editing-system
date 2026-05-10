# 계층: API 계층
# 역할: 모든 v1 엔드포인트 라우터를 하나로 모아서 main.py에 등록
#       개별 라우터 파일(projects, shorts, system)에 URL 접두사와 태그를 부여
# 의존: projects.py, shorts.py, system.py
# MVA 원칙: 라이터 집합을 한 곳에서 관리 -> 새 엔드포인트 추가 시 이 파일에 한 줄 추가

"""
API v1 라우터 집합
"""

from fastapi import APIRouter

# 각 도메인별 라우터 모듈 import
from app.api.v1 import projects, shorts, system, heatmap

# 최상위 v1 라우터 (main.py에서 /api/v1 접두사로 등록됨)
api_router = APIRouter()

# prefix: URL 경로 접두사 -> /api/v1/projects/...
# tag: Swagger UI에서 그룹 구분에 사용
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(shorts.router, prefix="/shorts", tags=["shorts"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(heatmap.router, prefix="/heatmap", tags=["heatmap"])