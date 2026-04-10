# 계층: API 계층 (Controller)
# 역할: 쇼츠 클립 관련 HTTP 엔드포인트
# 의존: AnalysisService, EditingService (DI로 주입 받음)
# MVA 원칙: 비즈니스 로직 없이 서비스에 위임

"""
쇼츠 클립 엔드포인트
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_analysis_service, get_editing_service
from app.schemas.api import ShortResponse, ShortsListResponse
from app.services.analysis_service import AnalysisService
from app.services.editing_service import EditingService

# APIRouter() 괄호 필수
# router = APIRouter    <- 클래스 자체를 할당
# router = APIRouter()  <- 인스턴스를 생성
router = APIRouter()

@router.get("/by-project/{project_id}", response_model=ShortsListResponse)
async def list_shorts(
    project_id: str,
    analysis_svc: AnalysisService = Depends(get_analysis_service),
):
    """
    프로젝트 별 쇼츠 목록 조회

    GET /api/v1/shorts/by-project/{project_id}
    Response: {"items": [...], "total": 5}
    """
    
    items = await analysis_svc.get_shorts_by_project(project_id)
    return ShortsListResponse(
        items=[
            ShortResponse(
                **s.__dict__,
                duration_sec=s.end_sec - s.start_sec,   # 계산 필드: 구간 길이
            )
            for s in items
        ],
        total=len(items)
    )

@router.post("/{shorts_id}/edit")
async def edit_shorts(
    shorts_id: str,
    editing_svc: EditingService = Depends(get_editing_service)
):
    """
    쇼츠 편집 시작 (리프레이밍 + 자막 + 인코딩)

    POST /api/v1/shorts/{shorts_id}/edit
    현재는 501 Not Implemented 반환 (2주차 구현 예정)
    """
    
    try:
        await editing_svc.reframe_clip(shorts_id)
        return {"message": "편집 완료"}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
