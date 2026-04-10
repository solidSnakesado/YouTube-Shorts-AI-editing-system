# 계층: 데이터 접근 계층 (Repository)
# 역할: 모든 레포지토리의 부모 클래스, 공동 CRUD 작업을 제네릭으로 구현하여
#       개별 레포지토리에서 반복 코드를 작성하지 않도록 한다.
# 의존: SQLAlchemy AsyncSession (DI로 주입받음)
# MVA 원칙: 제네릭 레포지토리 패턴 - 개발 속도 가속화
#
# 사용법:
#   class UserRepo(BaseRepository[User]):
#       def __init__(self, db):
#           super().__init__(User, db)
#       # create, get_by_id, get_all, update, delete가 자동으로 사용 가능
#       # 필요한 특수 쿼리만 추가하면 됨
"""
제네릭 레포지토리 베이스 클래스

공동 CRUD 작업을 자동화하여 개발 시간을 단축한다.
"""

# Generic: 타입 파라미터를 받는 제네릭 클래스 정의용
# TypeVar: 제네릭 타입 변수 (T와 비슷한 개념)
from typing import Generic, Optional, TypeVar, Type

# select: SQL SELECT 쿼리 빌더
# func: SQL 함수 (COUNT, SUM 등) 호출용
from sqlalchemy import select, func

# AsyncSession: 비동기 DB 세션 (모든 쿼리가 await로 실행됨)
from sqlalchemy.ext.asyncio import AsyncSession

# SQLModel: 이 프로젝트의 모든 도메인 모델의 부모 클래스
from sqlmodel import SQLModel

# ModelType: 제네릭 타입 변수
# bound=SQLModel: SQLModel을 상속한 클래스만 허용
# ex> BaseRepository[Project]에서 ModelType = Project
ModelType = TypeVar("ModelType", bound=SQLModel)

class BaseRepository(Generic[ModelType]):
    """
    제네릭 CRUD 레포지토리

    Python의 Generic을 활용하여 어떤 SQLModel이든 처리할 수 있는 범용 CRUD 메서드를 제공

    제공 메서드:
        create()        - 엔티티 생성 (INSERT)
        get_by_id()     - ID로 단건 조회 (SELECT WHERE id=?)
        get_all()       - 전체 목록 조회 (SELECT with OFFSET/LIMIT)
        count()         - 전체 건수 조회 (SELECT COUNT(*))
        update()        - 엔티티 부분 수정 (UPDATE)
        delete()        - 엔티티 삭제 (DELETE)
    """

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        Args:
            model: 대상 SQLModel 클래스 (ex> Project, Shorts) 
            db: 비동기 DB 세션 (DI로 주입받음)
        """

        self.model = model              # SELECT, INSERT 등에서 사용할 테이블 클래스
        self.db = db                    # DB 세션 (이 세션을 통해 모든 쿼리 실행)

    async def create(self, obj: ModelType) -> ModelType:
        """
        엔티티 생성 (INSERT)

        flush()로 SQL을 즉시 실행하되, commit은 하지 않는다.
        commit은 get_db()의 세션 관리자가 요청 종료 시 일관 처리한다.
        """

        self.db.add(obj)                # 세션의 변경 추적 대상에 등록
        await self.db.flush()           # SQL 실행 (INSERT INTO ...) - 아직 commit 아님
        await self.db.refresh(obj)      # DB에서 최신 값을 다시 읽어 객체에 반영
        return obj
    
    async def get_by_id(self, id: str) -> Optional[ModelType]:
        """
        ID로 단건 조회

        결과가 없으면 None 반환 (scalar_one_or_none)
        결과가 2개 이상이면 예외 발생 (primary key이므로 분가능하지만 안전장치로 지정)
        """

        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
            # 생성되는 SQL: SELECT * FROM projects WHERE id = ?
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """
        전체 목록 조회 (페이징 지원)

        Args:
            skip: 건너뛸 레코드 수 (OFFSET)
            limit: 가져올 최대 레코드 수 (LIMIT)

        created_at 내림차순: 최신 데이터가 먼저 나오도록
        """

        result = await self.db.execute(
            select(self.model)
            .offset(skip)
            .limit(limit)
            .order_by(self.model.created_at.desc())
            # SQL: SELECT * FROM projects ORDER BY created_at DESC OFFSET ? LIMIT ?
        )
        # scalars(): Row 객체에서 첫 번째 칼럼(모델 인스턴스)만 추출
        # all(): 모든 결과를 리스트로 반환
        return list(result.scalars().all())
    
    async def count(self) -> int:
        """
        전체 건수 조회

        UI 페이지네이션에서 "총 N건" 표시에 사용
        """

        result = await self.db.execute(
            select(func.count()).select_from(self.model)
            # SQL: SELECT COUNT(*) FROM projects
        )
        # 결과가 반드시 1개이므로 scalar_one 사용 
        return result.scalar_one()
    
    async def update(self, obj: ModelType, data: dict) -> ModelType:
        """
        엔티티 부분 업데이트

        전달된 dict의 key-value만 객체에 반영
        존재하지 않는 속성은 무시 (hasattr 체크)

        Args:
            obj: 업데이트할 엔티티 인스턴스 (이미 DB에서 조회된)
            data: 변견할 필드와 값 (ex> {"status": "completed", "title": "..."})
        """

        for key, value in data.items():
            if hasattr(obj, key):           # 모델에 존재하는 필드만 업데이트
                setattr(obj, key, value)    # obj.status = "completed"과 동일

        self.db.add(obj)                    # 변경된 객체를 세션에 다시 등록
        await self.db.flush()               # UPDATE SQL 실행
        await self.db.refresh(obj)          # DB 최신 값 반영

        return obj
    
    async def delete(self, obj: ModelType) -> None:
        """
        엔티티 삭제

        Args:
            obj: 삭제할 엔티티 인스턴스
        """
        
        await self.db.delete(obj)           # DELETE FROM ... WHERE id = ?
        await self.db.flush()               # SQL 실행