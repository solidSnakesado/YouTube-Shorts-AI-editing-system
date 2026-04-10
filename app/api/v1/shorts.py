"""
쇼츠 클립 엔드포인트
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_analysis_service, get_editing_service
from app.schemas.api import ShortResponse, ShortsListResponse
from app.services.analysis_service import AnalysisService
from app.services.editing_service import EditingService

router = APIRouter()

@router.get("/by-project/{project_id}", response_model=ShortsListResponse)
async def list_shorts(
    project_id: str,
    analysis_svc: AnalysisService = Depends(get_analysis_service),
):
    items = await analysis_svc.get_shorts_by_project(project_id)
    return ShortsListResponse(
        items=[
            ShortResponse(
                **s.__dict__,
                duration_sec=s.end_sec - s.start_sec,
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
    try:
        await editing_svc.reframe_clip(shorts_id)
        return {"message": "편집 완료"}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
