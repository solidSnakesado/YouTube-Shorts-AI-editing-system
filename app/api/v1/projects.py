# 계층: API 계층 (Controller)
# 역할: HTTP 요청을 받아 서비스 계층에 위임하고, 응답을 반환
#       비즈니스 로직, DB 접근 코드를 이곳에 작성하지 않는다.
# 의존: VideoService, AnalysisService (DI로 주입받음)
# MVA 원칙: API 계층은 요청/응담 반환만 담당, 로직은 서비스로 위임
#
# 3~5일차 변경사항:
#   - POST /{project_id}/transcribe 엔드포인트 추가 (Whisper ASR)
#   - POST /{project_id}/analyze 엔드포인트에 전사 -> 하이라이트 2단계 흐름 반영

"""
프로젝트 엔드포인트

HTTP 요청/응답만 담당하며, 비즈니스 로직은 서비스 계층으로 위임한다.
"""

from fastapi import APIRouter, Depends, HTTPException, status

# DI 체인에서 서비스 인스턴스를 가져오는 팩토리 함수
from app.core.dependencies import get_video_service, get_analysis_service

# 요청/응답 스키마 (Pydantic 모델)
from app.schemas.api import ProjectCreate, ProjectResponse, ProjectListResponse

# 서비스 클래스 (타입 힌트용)
from app.services.video_service import VideoService
from app.services.analysis_service import AnalysisService

# APIRouter 인스턴스: router.py 에서 api_router 에 등록됨
router = APIRouter()

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,

    # Depends(get_video_service): FastAPI가 DI 체인을 자동 해석하여
    # get_db -> get_projet_repo -> get_video_service 순서로 객체 생성 후 주입
    video_svc: VideoService = Depends(get_video_service),
):
    """
    유투브 URL로 새 프로젝트 생성

    POST /api/v1/projects/
    Body: {"youtube_url": "https://...", "max_shorts": 5, "short_duration_sec": 60}
    Response: 201 Created + ProjectResponse

    body는 ProjectCreate 스키마로 자동 검증됨:
        - youtube_url이 유효한 URL 인지
        - max_shorts가 1~20범위 인지
        - shorts_duration_sec이 15~180 범위인지
    -> 검증 실해 시 FastAPI가 자동으로 422 Validation Error 반환
    """
    
    project = await video_svc.create_project(str(body.youtube_url))
    return ProjectResponse(
        **project.__dict__,     # SQLModel 객체의 모든 필드를 dict로 변환하여 전달
        shorts_count=0          # 새프로젝트이므로 쇼츠 수는 0
    )

@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    skip: int = 0,              # 쿼리 파라미터: ?skip=10 -> 10개 건너뛰기 (OFFSET)
    limit: int = 20,            # 쿼리 파라미터: ?limit=5 -> 5개만 가져오기 (LIMIT)
    video_svc: VideoService = Depends(get_video_service),
):
    """
    프로젝트 목록 조회

    GET /api/v1/projects/?skip=0&limit=20
    Response: {"items": [...], "total": 42}
    """
    
    items, total = await video_svc.list_projects(skip=skip, limit=limit)

    return ProjectListResponse(
        items=[
            ProjectResponse(
                **p.__dict__, 
                shorts_count=len(getattr(p, "shorts", []))  # 쇼츠가 로드되지 않았다면 빈 리스트
            )
            for p in items
        ],
        total=total,
    )

@router.get("/{project_id}", response_model= ProjectResponse)
async def get_project(
    project_id: str,        # URL 경로 파라미터: /api/v1/projects/{project_id}
    video_svc: VideoService = Depends(get_video_service)
):
    """
    프로젝트 상세 조회 (관련 쇼츠 포함)

    GET /api/v1/projects/6b91314e-937a-4c2f-857c-571ba39f1d8d
    Response: ProjectResponse (shorts_count 포함)
    """
    
    project = await video_svc.get_project(project_id)

    if not project:
        # 404: 해당 ID의 프로젝트가 존재하지 않음
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
    """
    영상 다운로드 시작

    POST /api/v1/projects/{id}/download
    서비스에서 yt-dlp 다운로드 -> FFmpeg 오디오 추출 -> 메타데이터 추출을 수행
    """
    
    project = await video_svc.download_video(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
    
    return ProjectResponse(**project.__dict__, shorts_count=0)

@router.post("/{project_id}/transcribe", response_model=ProjectResponse)
async def transcribe_video(
    project_id: str,
    analysis_svc: AnalysisService = Depends(get_analysis_service),
    video_svc: VideoService = Depends(get_video_service)
):
    """
    음성 전사 실행 (3~5일차 신규 엔드포인트)

    POST /api/v1/projects/{id}/transcribe

    파이프라인에서의 위치:
        download -> **transcribe** -> analyze (하이라이트 추출)

    faster-whisper를 사용하여 오디오를 텍스트로 변환하고, 
    단어 단위 타임스탬프를 포함한 결과를 project.transcript_json에 저장

    전제 조건:
        - 다운로드가 완료되어 project.audio_path에 WAV 파일이 존재해야 함
        - project.status가 ANALYZING 상태여야 함 (download 완료 후 자동 전환)

    응답:
        - 성공: ProjectResponse (transcript_json이 채워진 상태)
        - 실패: 404 (프로젝트 미존재) 또는 500 (전사 실패)
    """  

    # 프로젝트 존재 확인
    project = await video_svc.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")
    
    # 전사 실행
    update_project = await analysis_svc.transcribe(project_id)

    if not update_project:
        raise HTTPException(
            status_code=500,
            detail="음성 전사에 실패했습니다. 프로젝트 상태를 확인하세요."
        )
    
    return ProjectResponse(
        **update_project.__dict__,
        shorts_count=0,
    )

@router.post("/{project_id}/analyze")
async def analyze_video(
    project_id: str,
    max_shorts: int = 5,
    analysis_svc: AnalysisService = Depends(get_analysis_service),
):
    """
    하이라이트 구간 추축 (LLM 기반)

    POST /api/v1/projects/{id}/analyze?max_shorts=5

    파이프라인에서의 위치:
        download -> transcribe -> **analyze** 

    전제 조건:
        - 전사가 완료되어 project.transcript_json이 존재해야 함

    현재 상태: 6~7일차 구현 예정 (501 반환)
    """
    
    try:
        shorts = await analysis_svc.extract_highlights(project_id, max_shorts)
        return {"message": "분석 완료", "short_count": len(shorts)}
    except NotImplementedError as e:
        # 501: 아직 구현되지 않은 기능
        raise HTTPException(status_code=501, detail=str(e))