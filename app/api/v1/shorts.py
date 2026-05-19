# 계층: API 계층 (Controller)
# 역할: 쇼츠 클립 관련 HTTP 엔드포인트
# 의존: AnalysisService, EditingService (DI로 주입 받음)
# MVA 원칙: 비즈니스 로직 없이 서비스에 위임
#
# 8~10일차 변경사항:
#   - POST /{shorts_id}/edit: 501 스텁 -> 실제 리프레이밍 동작
#   - aspect_ratio 쿼리 파라미터 추가
# 11~12일차 변경사항:
#   - POST /{shorts_id}/subtitle:   자막 생성 엔드포인트 신규
#   - POST /{shorts_id}/encode  :   최종 인코딩 엔트포인트 신규
# 21일차 변경사항:
#   - POST /{shorts_id}/resize:     리사이징 엔드포인트 신규
#   - edit 엔드포인트 독스트링에 다양한 종횡비 지원 명시

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
    쇼츠 리프레이밍 실행

    POST /api/v1/shorts/{shorts_id}/edit?aspect_ratio=9:16

    지원 종횡비: 9:16(쇼츠), 16:9(가로), 1:1(정사각), 4:5, 4:3, 16:10
    YOLOv8 피사체 추적 -> 스무딩 -> 적응형 크롭 -> FFmpeg 리프레이밍
    """

    result = await editing_svc.reframe_clip(shorts_id, aspect_ratio)

    if not result:
        raise HTTPException(status_code=500, detail="리프레이밍에 실패했습니다.")
    
    return ShortResponse(
        **result.__dict__,
        duration_sec=round(result.end_sec - result.start_sec, 3)
    )

@router.post("/{shorts_id}/subtitle", response_model=ShortResponse)
async def subtitle_shorts(
    shorts_id: str,
    editing_svc: EditingService = Depends(get_editing_service),
):
    """
    동적 자막 생성 - 11~12일차 구현

    POST /api/v1/shorts/{shorts_id}/subtitle

    파이프라인에서의 위치: edit(리프레이밍) -> **subtitle** -> encode

    Whisper 단어 타임스탬프 기반 ASS 포맷 자막 생성
    현재 발화 단어를 카라오케 스타일로 강조 (노란색, 110% 크기)
    FFmpeg ass 필터로 영상에 하드 서브 합성

    전제 조건:
        - edit(리프레이밍) 완료되어 output_path에 9:16 영상이 존재
        - 프로젝트에 transcript_json(전사 데이터)이 존재

    응답:
        - 성공: ShortResponse (output_path가 자막 합성 영상으로 업데이트)
        - 실패: 500 (자막 생성 실패)
    """

    result = await editing_svc.generate_subtitles(shorts_id)
    if not result:
        raise HTTPException(status_code=500, detail="자막 생성에 실패했습니다.")
    
    return ShortResponse(
        **result.__dict__,
        duration_sec=round(result.end_sec - result.start_sec, 3),
    )

@router.post("/{shorts_id}/encode", response_model=ShortResponse)
async def encode_shorts(
    shorts_id: str,
    editing_svc: EditingService = Depends(get_editing_service),
):
    """
    최종 인코딩 - 11~12일차 구현

    POST /api/v1/shorts/{shorts_id}/encode

    파이프라인에서의 위치: subtitle -> **encode** -> COMPLETED

    NVENC H.264 GPU 인코딩 + 오디오 노멀라이즈(-14 LUFS)
    + 피치 보존 속도 조정(1.05x) + movflags +faststart
    최종 출력: outputs/{shorts_id}.mp4 (1080x1920, H.264, AAC)

    전제 조건:
        - subtitle 완료되어 output_path에 자막 합성 영상이 존재

    응답:
        - 성공: ShortResponse (status=completed, output_path=최종 파일)
        - 실패: 500 (인코딩 실패) 
    """

    result = await editing_svc.encode_final(shorts_id)
    if not result:
        raise HTTPException(status_code=500, detail="최종 인코딩에 실패했습니다.")
    
    return ShortResponse(
        **result.__dict__,
        duration_sec=round(result.end_sec - result.start_sec, 3)
    )

@router.post("/{shorts_id}/resize", response_model=ShortResponse)
async def resize_shorts(
    shorts_id: str,
    aspect_ratio: str = "16:9",
    editing_svc: EditingService = Depends(get_editing_service),
):
    """
    기존 영상을 다른 종횡비로 리사이징 (21일차 신규)

    POST /api/v1/shorts/{shorts_id}/resize?aspect_ratio=16:9

    편집 완료된 영상을 레터박스 방식으로 다른 비율로 변환
    지원 종횡비: 9:16, 16:9, 1:1, 4:5, 4:3, 16:10
    """

    result = await editing_svc.resize_clip(shorts_id, aspect_ratio)
    if not result:
        raise HTTPException(status_code=500, detail="리사이징에 실패했습니다.")
    
    return ShortResponse(
        **result.__dict__,
        duration_sec=round(result.end_sec - result.start_sec, 3),
    )