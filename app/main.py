# 계층: 진입점 (Entry Point)
# 역할: FastAPI 인스턴스를 생성하고, 미들웨어와 라우터를 등록
#       이 파일에는 비즈니스 로직을 절대 작성하지 않는다.
#       "최소한의 코드"만 포함하는 것이 MVA 원칙
# 의존: api_router, settings, init_db
# MVA 원칙: main.py 모놀리스 방지 - 진입점은 등록만 담당

"""
YT Shorts AI - 애플리케이션의 집입점

최소한의 코드만 포함: FastAPI 인스턴스 생성, 미들웨어 등록, 라우터 마운트
비즈니스 로직은 절대 이곳에 작성하지 않는다.
"""

# asynccontextmanager: 비도ㅓㅇ기 컨텍스트 매니저 생성용 데코레이터
# FastAPI의 lifespan 이벤트(서버 시작/종료)를 관리하는 데 사용
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles     # 24일차: output/ 정적 파일 서빙 (쇼츠 재생용)
# CORS 미들웨어: 브라우저의 Cross-Origin 요청을 허용
# 프론트엔드(React 등)가 다른 포트에서 API를 호출할 때 필요
from fastapi.middleware.cors import CORSMiddleware

# v1 라우터 집합 (모든 엔드포인트가 여기에 등록)
from app.api.v1.router import api_router
# 환결 설정 (앱 이름 등)
from app.core.config import settings
# DB 테이블 초기화 함수
from app.core.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리.

    서버 시작 시 (yield 이전):
        - DB 테이블 자동 생성 (CREATE TABLE IF NOT EXISTS)
        - 향후: AI 모델 사전 로드, 캐시 워밍업 등

    서버 종료 시 (yield 이후)
        - 향후: 모델 언로드, 임시 파일 정리, DB 연결 풀 해제 등

    - lifespan은 FastAPI 0.93+에서 on_startup/on_shoutdown을 대체하는 방식
    """

    # - Startup -
    await init_db()     # SQLModel 메타데이터 기반 테이블 생성
    yield

    # - Shoutdown -
    pass                # 2주차에 리소스 정리 코드 추가 예정

# --------------------------------------------------------------
# FastAPI 앱 인스턴스 생성
# --------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,                                    # Swagger UI 제목: "YT Shorts AI"
    version="0.1.0",                                            # API 버전
    description="로컬 GPU 가속 기반 유부브 쇼츠 자동 편집 AI",       # Swagger 설명             
    lifespan=lifespan,                                          # 생명 주기 관리자
    docs_url="/docs",                                           # Swagger UI 경로 (http://localhost:8000/docs)
    redoc_url="/redoc",                                         # ReDoc UI 경로 (대체 API 문서)       
)

# --------------------------------------------------------------
# 미들웨어 등록 
# --------------------------------------------------------------
# CORS: Cross-Origin Resource Sharing
# 프론트엔드가 localhost:3000에서 API(localhost:8000)를 호출할 때 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                        # 프로토타입: 모든 오리진 허용, 프로덕션에서는 [https://yourdomain.com]으로 제한 필요
    allow_credentials=True,                                     # 쿠키/인증 정보 포함 허용
    allow_methods=["*"],                                        # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],                                        # 모든 헤더 허용
)

# --------------------------------------------------------------
# 라우터 등록 
# --------------------------------------------------------------
# prefix="/api/v1": 모든 엔드포인트에서 /api/v1 접두사 부여
# 결과 URL 예시: /api/v1/projects/, /api/v1/shorts/, /api/v1/system/status
app.include_router(api_router, prefix="/api/v1")
# output/ 디렉토리를 /static/outputs 로 서빙 (24일차: 쇼츠 재생 기능)
app.mount("/static/outputs", StaticFiles(directory=str(settings.output_path)), name="outputs")
# 36일차: temp/ 를 /static/temp 로 서빙 (라벨링용 원본 영상 재생, Range 자동 지원)
app.mount("/static/temp", StaticFiles(directory=str(settings.temp_path)), name="temp")

# --------------------------------------------------------------
# 헨스 체크 엔드포인트
# --------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health_check():
    """
    서버 상태 확인

    GET /health
    Response: {"status": "healthy", "version": "0.1.0"}

    로드밸런서, 모니터링 도구, CI/CD 파이프라인에서
    서버가 작동 중인지 확인하는 데 사용
    """
    return {"status": "healthy", "version": "0.1.0"}