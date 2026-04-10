"""
프로젝트 엔드포인트

HTTP 요청/응답만 담당하며, 비즈니스 로직은 서비스 계층으로 위임한다.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_video_service, get_analysis_service
from app.schemas.api import ProjectCreate, ProjectResponse, ProjectListResponse
from app.services.video_service import VideoService
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    video_svc: VideoService = Depends(get_video_service),
):
    project = await video_svc.create_project(str(body.youtube_url))
    return ProjectResponse(**project.__dict__, shorts_count=0)

@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    skip: int = 0,
    limit: int = 20,
    video_svc: VideoService = Depends(get_video_service),
):
    items, total = await video_svc.list

    return ProjectListResponse(
        items=[
            ProjectResponse(**p.__dict__, shorts_count=len(getattr(p, "shorts", [])))
            for p in items
        ],
        total=total,
    )

@router.get("/{project_id}", response_model= ProjectResponse)
async def get_project(
    project_id: str,
    video_svc: VideoService = Depends(get_video_service)
):
    project = await video_svc.get_project(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")
    
    return ProjectResponse(
        **project.__dict__,
        shorts_count=len(project.shorts) if project.shorts else 0,
    )

@router.post("/{project_id}/download", response_model=ProjectResponse)
async def download_video(
    project_id: str,
    video_svc: VideoService = Depends(get_video_service)
):
    project = await video_svc.download_video(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
    
    return ProjectResponse(**project.__dict__, shorts_count=0)

@router.post("/{project_id}/analyze")
async def analyze_video(
    project_id: str,
    max_shorts: int = 5,
    analysis_svc: AnalysisService = Depends(get_analysis_service),
):
    try:
        shorts = await analysis_svc.extract_highlights(project_id, max_shorts)
        return {"message": "분석 완료", "short_count": len(shorts)}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))