# 계층: 인프라 계층 (Core)
# 역할: 비동기 데이터베이스 연결 엔진과 세션 팩토리를 관리
#       모든 DB 작업은 이 모듈이 제공하는 세션을 통해 수행되며,
#       세션의 생명주기 (생성 -> 커밋 -> 롤백 -> 종료)를 자동으로 관리한다.
# 의존: app.core.config (DATABASE_URL, is_dev 설정값)
# MVA 원칙: DB 접근을 한 곳에서 관리하여, DB 기술 교체 시 이 파일만 수정
#
# 11~12일차 변경사항:
#   - get_db(): HTTPException은 커밋 유지, 시스템 에러만 롤백
#     -> 서비스에서 FAIDED 상태 설정 후 API에서 HTTPException 발생 시
#        세션이 롤백되어 FAIDED가 DB에 반영되지 않던 버그 수정

"""
데이터베이스 연결 및 세션 관리

비동기 SQLAlchemy 엔진을 사용하며, 
의존성 주입을 통해 각 요청에 독립적인 세션을 제공한다.
"""

# AsyncGenerator: yield를 사용하는 비동기 제너레이터의 타입 힌트
from collections.abc import AsyncGenerator

# AsyncSession: 비동기 DB 세션 (await 쿼리 실행)
# async_sessionmaker: AsyncSesstion을 생성하는 팩토리 클래스
# create_async_engine: 비동기 DB 연결 엔진 생성 함수
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
# SQLModel: SQLAlchemy + Pydantic 통합 ORM
# SQLModel.matadata: 모든 테이블 정즤 정보를 담고 있는 메타데이터 객체
from sqlmodel import SQLModel

# HTTPException: FastAPI의 의도적 에러 으답 (비즈니스 로직 판단)
# 시스템 장애와 구분하여 세션 커밋/롤백 전략을 분리하기 위해 import
from fastapi import HTTPException

# 환경 설정에서 DB URL 등을 가져옴
from app.core.config import settings

# --------------------------------------------------------------
# 엔진 생성
# --------------------------------------------------------------
# 엔진 = DB 서버와의 연결 풀 (Connection Pool)을 관리하는 객체
# 앱 전체에서 하나만 존재하며, 여러 세션이 이 엔진을 공유
engine = create_async_engine(
    settings.DATABASE_URL,      # "sqllite+aiosqlite:///./data/shorts_ai.db"
    echo=settings.is_dev,       # True이면 실행되는 SQL을 콘솔에 출력 (디버그용) 
    future=True,                # SQLAlchemy 2.0 스타일 API 사용
)

# --------------------------------------------------------------
# 세션 팩토리 생성
# --------------------------------------------------------------
# 세션 팩토리: 호출할 때마다 새로운 AsyncSession 인스턴스를 생성
# 각 API 요청마다 독립적인 세션을 사용하여 트랜잭션 격리를 보장
async_session_factory = async_sessionmaker(
    engine,                     # 위에서 생성한 엔진과 연결
    class_=AsyncSession,        # 생성할 세션의 클래스 (비동기)
    expire_on_commit=False,     # 커밋 후에도 객체 속성에 접근 가능, False가 아니면 커밋 후 project.title 같은 접근 시 재조회 필요
)

async def init_db() -> None:
    """
    데이터베이스 테이블 초기화

    SQLModel에 등록된 모든 모델(table=True)의 테이블을 생성
    이미 존재하는 테이블은 건너뛴다 (CREATE TABLE IF NOT EXISTS).

    호출 시점:
        - app/main.py의 lifespan 이벤트 (서버 시작 시)
        - tests/unit/test_api.py의 fixture (테스트 시작 시)

    프로덕션에서는 Alembic 마이그레이션으로 대체해야 한다.
    """
    async with engine.begin() as conn:
        # run_sync: 동기 하수를 비동기 컨텍스트에서 실행
        # metadata.create_all: 모든 테이블 DDL을 실행
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    DB 세션 의존성 (Dependency Injecttion용 제너레이터)

    FastAPI의 Depends()와 함께 사용되며, 각 API 요청 마다
    독립적인 DB 세션을 생성하고, 요청 종료 시 자동으로 정리

    커밋/롤백 전략 (11~12일차 수정):
        - HTTPException: 의도적 비즈니스 에러 -> 커밋 유지
          (서비스가 FAILED 상태를 설정한 후 APIㅁ가 500을 반환하는 경우,
           FAIDED 상태가 DB에 반영되어야 함)
        - 그 외 Exception: 예상치 못한 시스템 장애 -> 롤백
          (DB 오류, 네트워크 장애 등 데이터 무결성 보호)
    
    동작 흐름:
        1. 요청 도착 -> 새 세션 생성
        2. yield session -> 엔드포인트에서 세션 사용
        3. 정상 완료 -> commit (변경사항 DB에 반영)
        4. HTTPException -> commit (비즈니스 판단 보존) -> 재전파
        5. 기타 Exception -> rollback (데이터 무결성 보호) -> 재전파
        6. finally -> cloase (세션 자원 해제)

        주의: 이 함수는 레포지토리 계층에 주입되며,
              API 엔드포인트에서 직접 사용하지 않는다 (MVA 계층 규칙)
    """
    async with async_session_factory() as session:
        try:
            yield session               # 이 시점에서 엔드포인트 코드가 실행됨
            await session.commit()      # 예외 없으면 변경사항 커밋
        except HTTPException:
            # HTTPException은 의도적 비즈니스 판단 (ex> 서비스에서 FAIDED 설정 후 500 반환)
            # 서비스가 flush()한 변경사항(FAILED 상태 등)을 DB에 반영해야 하므로 커밋
            await session.commit()
            raise                       # HTTPException을 상위로 전파 (FastAPI가 응답 생성)
        except Exception:
            await session.rollback()    # 예외 발생 시 모든 변경사항 롤백
            raise                       # 예외를 상위로 전파 (FastAPI가 500 응답 생성)
        finally:
            await session.close()       # 세션 자원 해제 (DB 연결을 풀에 반환)