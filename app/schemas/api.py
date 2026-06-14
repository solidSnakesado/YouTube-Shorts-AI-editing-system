# 계층: API 스키마 계층 (Schemas)
# 역할: API 요청(Request)과 응답(Response)의 데이터 형식을 정의 한다
#       도메인 모델(model/domain.py)과 분리하여
#       DB 스키마 변경이 API 계약에 영향을 주지 않도록 한다
# 의존: 없음 (순수 pYDANTIC 모델)
# MVA 원칙: 관심사 분리 - DB ≠ 스키마 
#
# 왜 도메인 모델과 분리하는 가?
#   - DB에는 source_path, audio_path 같은 내부 경로가 있지만 API 응답에 노출하면 안됨
#   - API에는 shorts_count 같은 계산 필드가 필요하지만 DB에는 없음
#   - DB 컬럼 추가/삭제 시 API 응답 형식을 그대로 유지할 수 있음
"""
API 스키마 정의

Pydantic 모델로 요청/응답 형식을 정의
"""

from datetime import datetime
from typing import Literal, Optional

# BaseModel: Pydantic 기반 데이터 검증 클래스
# Field: 필드 제약조건 정의 (최소값, 최대값, 기본값 등)
# HttpUrl: URL 형식 자동 검증 ("https://..." 형태인지 확인)
from pydantic import BaseModel, Field, HttpUrl

# --------------------------------------------------------------
# Project 요청/응답 스키마
# --------------------------------------------------------------

class ProjectCreate(BaseModel):
    """
    프로젝트 생성 요청 스키마
    13일차 변경: shorts_duration_sec 제거 - LLM이 콘텐트레 맞게 자동 판단
    """

    youtube_url: HttpUrl                                            # Pydantic이 URL 형식을 자동 검증 (잘못된 URL 이면 422 에러)
    max_shorts: int = Field(default=5, ge=1, le=20)                 # ge=1: 최소 1개, le=20: 최대 20개 -> 범위 밖이면 자동으로 422 Validation Error

class ProjectResponse(BaseModel):
    """
    프로젝트 응답 스키마

    GET /api/v1/projects/{id} 등의 응답 형식
    도메인 모델의 모든 필드를 노출하지 않고, API에 필요한 것만 선별한다
    """

    id: str
    youtube_url: str
    title: Optional[str]                # 다운로드 전에는 None
    duration_sec: Optional[float]       # 다운로드 전에는 None
    status: str                         # "pending", "downloading", "completed" 등
    error_message: Optional[str]        # 실패 시에만 값이 있음
    shorts_count: int = 0               # 생성된 쇼츠 수 (DB 칼럼이 아닌 계산 필드)
    created_at: datetime
    updated_at: datetime


    # from_attributes=True: SQLModel 객체를 dict 변환 없이 바로 매핑 가능
    # ex> ProjectResponse(**project.__dict__) 가 작동하려면 필요
    model_config = {"from_attributes": True}

class ProjectListResponse(BaseModel):
    """
    프로젝트 목록 응답, 페이징 처리를 위한 total 포함.
    """

    items: list[ProjectResponse]    # 프로젝트 배열
    total: int                      # 전체 건수 (UI 페이지네이션 용)

# --------------------------------------------------------------
# Shorts 요청/응답 스키마
# --------------------------------------------------------------

class ShortResponse(BaseModel):
    """
    쇼츠 클립 응답 스키마

    하이라이트 구간 정보와 편집 결과를 포함
    """

    id: str
    project_id: str
    status: str
    start_sec: float                        # 시작 시점 (초)
    end_sec: float                          # 끝 시점 (초)
    duration_sec: Optional[float]           # end - start 계산값 (API에서 편의 제공)
    highlight_reason: Optional[str]         # LLM의 선정 사유
    hook_score: Optional[float]             # 흥미도 점수 (0~1)
    output_path: Optional[str]              # 완성된 파일 경로
    title_suggestion: Optional[str]         # 제안 제목
    tags_suggestion: Optional[str]          # 제안 태그 (JSON 배열)
    feedback: Optional[str] = None          # 33일차: 사람 평가 (ok/no), UI 상태 표시용
    feedback_reason: Optional[str] = None   # 33일차: NO 사유 (selection/boundary/editing)
    created_at: datetime                

    model_config = {"from_attributes": True}

# 33일차: 피드백 제출 요청 (피드백 루프 C)
class FeedbackRequest(BaseModel):
    """쇼츠 피드백 제출 요청
    
    Literal 검증으로 허용값 외 입력은 422 자동 거부
    -> D(학습데이터 변환)에서 파싱 안정성 보장
    """

    feedback: Literal["ok", "no"]                                                   # 사람 평가
    feedback_reason: Optional[Literal["selection", "boundary", "editing"]] = None   # NO 사요 (선택)

class ShortsListResponse(BaseModel):
    """
    쇼츠 목록 응답
    """

    items: list[ShortResponse]
    total: int

# --------------------------------------------------------------
# 시스템 상태 스키마
# --------------------------------------------------------------

class GPUStatus(BaseModel):
    """
    GPU 상태 정보

    nvidia-smi 명령어 결과를 파싱하여 채움
    GPU가 없는 환경에서 available=False, 나머지는 None
    """

    available: bool                         # GPU 사용 가능 여부
    name: Optional[str] = None              # "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    vram_total_mb: Optional[int] = None     # 전체 VRAM (MB)
    vram_used_mb: Optional[int] = None      # 사용 중인 VRAM (MB)

class SystemStatus(BaseModel):
    """
    시스템 상태 응답

    GET /api/v1/system/status 의 응답 생성
    서버 헬스 체크와 GPU 모니터링에 사용
    """
    status: str                     # "healthy"
    gpu: GPUStatus                  # GPU 상세 정보
    models_loaded: list[str] = []   # 현재 GPU에 로드된 모델 목록 (2주차 구현)