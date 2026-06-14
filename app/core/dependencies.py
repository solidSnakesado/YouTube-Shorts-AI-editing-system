# 계층: 인프라 계층 (Core)
# 역할: 의존성 주입(DI) 체인을 정의
#       FastAPI의 Depends() 시스템이 이 체인을 자동으로 해석하여
#       DB 세션 -> 레포지토리 -> 서비스 순서로 객체를 생성/주입
# 의존: database.py, 모든 레포지토리, 모든 서비스
# MVA 원칙: DI 체인으로 계층 간 결합도를 낮추고, 테스트 시 교체를 용이하게 함
# ┌────────────────────────────────────────────────────────────────────────────┐
# |  DI 해석 순서 (FastAPI가 자동으로 수행)                                        |
# |                                                                            |
# |  요청 도착                                                                   |
# |      ├→ get_db() -> AsyncSession 생성                                       |
# |      |     └→ get_project_repo(db) -> ProjectRepository 생성                |
# |      |                └→ get_video_service(repo) -> VideoService 생성       |
# |      |                            └→ create_project(svc) -> 엔드포인트 실행  |
# |      |                                                                     |
# |      |                                                                     |
# |  요청 종료 -> 세션 자동 커밋/롤백/종료                                         |
# └────────────────────────────────────────────────────────────────────────────┘

"""
의존성 주입 체인 모듈

컨트롤러 -> 서비스 -> 레포지토리 -> DB 순서의 의존성 체인을 구성
"""

# Depends: FastAPI의 의존성 주입 함수, 파라미터에 선언하면 자동으로 호출됨
from fastapi import Depends

# AsyncSession: 비동기 DB 세션 타입 (타입 힌트용)
from sqlalchemy.ext.asyncio import AsyncSession

# DB 세션 제공 함수 (이 체인의 최하위)
from app.core.database import get_db

# 레포지토리 (데이터 접근 계층)
from app.repositories.project_repository import ProjectRepository
from app.repositories.shorts_repository import ShortsRepository

# 서비스 (비즈니스 로직 계층)
from app.services.video_service import VideoService
from app.services.analysis_service import AnalysisService
from app.services.editing_service import EditingService
from app.services.feedback_service import FeedbackService   # 33일차: 피드백 루프

# --------------------------------------------------------------
# 레포지토리 계층 (DB 세션을 주입받아 레포지토리 생성)
# --------------------------------------------------------------
# Depends(get_db): FastAPI가 get_db()를 먼저 호출하여 세션을 생성하고, 그 결과를 db 파라미터에 전달

def get_project_repo(db: AsyncSession = Depends(get_db)) -> ProjectRepository:
    """
    프로젝트 레포지토리 생성, DB 세션을 주입받아 초기화
    """
    return ProjectRepository(db)

def get_short_repo(db: AsyncSession = Depends(get_db)) -> ShortsRepository:
    """
    쇼츠 레포지토리 생성, DB 세션을 주입받아 초기화
    """
    return ShortsRepository(db)

# --------------------------------------------------------------
# 서비스 계층 (레포지토리를 주입받아 서비스 생성)
# --------------------------------------------------------------
# Depends(get_project_repo): FastAPI가 get_project_repo()를 먼저 호출하고, 그 결과(ProjectRepository 인스턴스)를 전달
# 즉, get_db -> get_project_repo -> get_video_service 순서로 체인이 해석

def get_video_service(project_repo: ProjectRepository = Depends(get_project_repo),) -> VideoService:
    """
    비디오 서비스 생성, 프로젝트 레포지토리를 주입받아 초기화
    """
    return VideoService(project_repo)

def get_analysis_service(
    project_repo: ProjectRepository = Depends(get_project_repo),
    shorts_repo: ShortsRepository = Depends(get_short_repo),
) -> AnalysisService:
    """
    분석 서비스 생성, 프로젝트 + 쇼츠 레포지토리를 주입받아 초기화
    """
    return AnalysisService(project_repo, shorts_repo)

def get_editing_service(
    shorts_repo: ShortsRepository = Depends(get_short_repo),
    project_repo: ProjectRepository = Depends(get_project_repo),
) -> EditingService:
    """
    편집 서비스 생성, 쇼츠 + 프로젝트 레포지토리를 주입받아 초기화
    8~10일차 변경: ProjectRepository 추가 (소스 영상 경로 조회용)
    """
    return EditingService(shorts_repo, project_repo)

# 33일차: 피드백 서비스 (피드백 루프 C)
def get_feedback_service(
    shorts_repo: ShortsRepository = Depends(get_short_repo),
) -> FeedbackService:
    """
    피드백 서비스 생성, 쇼츠 레포지토리를 주입받아 초기화
    """

    return FeedbackService(shorts_repo)