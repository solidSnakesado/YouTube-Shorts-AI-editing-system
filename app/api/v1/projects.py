# 계층: API 계층 (Controller)
# 역할: HTTP 요청을 받아 서비스 계층에 위임하고, 응답을 반환
#       비즈니스 로직, DB 접근 코드를 이곳에 작성하지 않는다.
# 의존: VideoService, AnalysisService (DI로 주입받음)
# MVA 원칙: API 계층은 요청/응답 반환만 담당, 로직은 서비스로 위임
#
# 3~5일차 변경사항:
#   - POST /{project_id}/transcribe 엔드포인트 추가 (Whisper ASR)
# 6~7일차 변경사항:
#   - POST /{project_id}/analyze 엔드포인트: 501 스텁 -> 실제 동작
#   - 응답 모델을 ShortListResponse로 변경 (생성한 쇼츠 목록 변환)
#   - ShortResponse import 추가

"""
프로젝트 엔드포인트

HTTP 요청/응답만 담당하며, 비즈니스 로직은 서비스 계층으로 위임한다.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

# DI 체인에서 서비스 인스턴스를 가져오는 팩토리 함수
from app.core.dependencies import get_video_service, get_analysis_service

# 요청/응답 스키마 (Pydantic 모델)
# ShortResponse, ShortListResponse: 6~7일차에 analyze 응답용으로 추가
from app.schemas.api import ProjectCreate, ProjectResponse, ProjectListResponse, ShortResponse, ShortsListResponse

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
    Body: {"youtube_url": "https://...", "max_shorts": 5}
    Response: 201 Created + ProjectResponse

    body는 ProjectCreate 스키마로 자동 검증됨:
        - youtube_url이 유효한 URL 인지
        - max_shorts가 1~20범위 인지
    -> 검증 실패 시 FastAPI가 자동으로 422 Validation Error 반환
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
                # 36일차: getattr가 미로드 관계에 lazy load 유발(async->MissingGreenlet). __dict__로 로드 여부만 확인
                shorts_count=len(p.__dict__["shorts"]) if "shorts" in p.__dict__ else 0
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

    GET /api/v1/projects/{project_id}
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
    quality: int = 1080,        # 38일차: 라벨링 480 / 발행 1080 (해상도)
    video_svc: VideoService = Depends(get_video_service)
):
    """
    영상 다운로드 시작

    POST /api/v1/projects/{id}/download?quality=480
    서비스에서 yt-dlp 다운로드 -> FFmpeg 오디오 추출 -> 메타데이터 추출을 수행
    38일차: quality 쿼리 파라미터로 해상도 지정 (기본 1080, 라벨링 480)
    """
    
    project = await video_svc.download_video(project_id, quality=quality)

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

    파이프라인에서의 위치: download -> **transcribe** -> analyze (하이라이트 추출)

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
        raise HTTPException(status_code=500, detail="음성 전사에 실패했습니다. 프로젝트 상태를 확인하세요.")
    
    return ProjectResponse(**update_project.__dict__, shorts_count=0)

@router.post("/{project_id}/analyze", response_model=ShortsListResponse)
async def analyze_video(
    project_id: str,
    max_shorts: int = 5,                        # 쿼리 파라미터: ?max_shorts=5
    target_duration_sec: Optional[int] = None,  # 쿼리 파라미터: ?target_duration_sec=30 (10~60초, 미입력 시 LLM 자동)
    analysis_svc: AnalysisService = Depends(get_analysis_service),
    video_svc: VideoService = Depends(get_video_service),
):
    """
    하이라이트 구간 추출 (LLM 기반) - 6~7일차 구현 / 13일차: LLM 자동 길이 판단
    24일차: target_duration_sec 파라미터 추가 (사용자 지정 쇼츠 길이)

    POST /api/v1/projects/{id}/analyze?max_shorts=5&target_duration_sec=30

    target_duration_sec: 10~60 사이 정수 입력 시 해당 길이 기준으로 쇼츠 생성
                         미 입력 시 LLM이 콘텐츠에 맞게 10~120초 범위에서 자동 결정
    """

    # target_duration_sec 범위 검증 (입력된 경우에만 처리)
    if target_duration_sec is not None and not (10 <= target_duration_sec <= 60):
        raise HTTPException(status_code=400, detail="target_duration_sec는 10~60 사이 정수여야 합니다.")

    # 프로젝트 존재 확인
    project = await video_svc.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")
    
    # 하이라이트 추출 실행 (서비스 계층에 위임)
    shorts = await analysis_svc.extract_highlights(project_id, max_shorts, target_duration_sec)
    if not shorts:
        raise HTTPException(status_code=500, detail="하이라이트 추출에 실패했습니다.")
    
    # Shorts 엔티티 -> ShortResponse 스키마 변환
    return ShortsListResponse(
        items=[
            ShortResponse(
                **s.__dict__,
                # duration_sec: DB 컬럼이 아닌 계산 필드 (API 편의용)
                duration_sec=round(s.end_sec - s.start_sec, 3),
            )
            for s in shorts
        ],
        total=len(shorts),
    )