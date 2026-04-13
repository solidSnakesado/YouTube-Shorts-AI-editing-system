# 계층: API 계층 (Controller)
# 역할: 쇼츠 클립 관련 HTTP 엔드포인트
# 의존: AnalysisService, EditingService (DI로 주입 받음)
# MVA 원칙: 비즈니스 로직 없이 서비스에 위임
#
# 8~10일차 변경사항:
#   - POST /{shorts_id}/edit: 501 스텁 -> 실제 리프레이밍 동작
#   - aspect_ratio 쿼리 파라미터 추가
#   - ShortResponse 응답 반환으로 변경

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
                duration_sec=s.end_sec - s.start_sec,
            )
            for s in items
        ],
        total=len(items)
    )

@router.post("/{shorts_id}/edit", response_model=ShortResponse)
async def edit_shorts(
    shorts_id: str,
    aspect_ratio: str = "9:16",                 # 쿼리 파라미터: ?aspect_ratio=9:16
    editing_svc: EditingService = Depends(get_editing_service)
):
    """
    쇼츠 리프레이밍 실행 - 8~10일차 구현 완료

    POST /api/v1/shorts/{shorts_id}/edit?aspect_ratio=9:16

    파이프라인에서의 위치: analyze -> **edit** (리프레이밍)

    YOLOv8로 피사체 추적 -> 카마라 스무딩 -> 적응형 크롭 전략 -> 
    FFmpeg로 16:9 -> 9:16 리프레이밍

    전제 조건:
        - analyze 완료되어 Shorts 엔티티가 DB에 존재
        - 프로젝트의 sourcr_path에 소스 영상 파일이 존재

    응답:
        - 성공: ShortResponse (output_path 포함)
        - 실패: 404 (쇼츠 미존재) 또는 500 (리프레이밍 실패)

    이전 상태: 501 Not Implemented (스텁)
    """

    result = await editing_svc.reframe_clip(shorts_id, aspect_ratio)

    if not result:
        raise HTTPException(status_code=500, detail="리프레이밍에 실패했습니다.")
    
    return ShortResponse(
        **result.__dict__,
        duration_sec=round(result.end_sec - result.start_sec, 3)
    )
