# 계층: 데이터 접근 계층 (Repository)
# 역할: Project 엔티티에 특화된 DB 조회/수정 메시드를 제공
#       공통 CRUD(create, get_by_id, get_all, update, delete)는 
#       부모 클래스 BaseRepository에서 상속받아 사용한다.
# 의존: BaseRepository, Project 모델
# MVA 원칙: 레포지토리 패턴 - 서비스 계층이 SQL을 직접 작성하지 않음
 
"""
프로젝트 레포지토리
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# selectinload: 관계된 엔티티를 한 번의 추가 쿼리로 미리 로드 (N+1 문제 방지)
# ex> Project를 조회할 때 관련 Shorts도 함께 로드
from sqlalchemy.orm import selectinload

from app.models.domain import Project, ProjectStatus
from app.repositories.base_repository import BaseRepository

class ProjectRepository(BaseRepository[Project]):
    """
    프로젝트 전용 레포지토리

    상속받는 메서드 (BaseRepository):
        create(project)         -> INSERT
        get_by_id(id)           -> SELECT WHERE id=?
        get_all(skip, limit)    -> SELECT with paging
        count()                 -> SELECT COUNT(*)
        update(project, data)   -> UPDATE
        delete(project)         -> DELETE

    이 클래스에서 추가하는 메서도:
        get_with_shorts()       -> 쇼츠 목록 포함 조회
        get_by_status()         -> 특정 상태의 프로젝트 목록
        update_status()         -> 상태 변경 헬퍼
    """

    def __init__(self, db: AsyncSession):
        # 부모 클래스에 모호델 타입(Project)과 DB 세션을 전달
        super().__init__(Project, db)

    async def get_with_shorts(self, project_id: str) -> Optional[Project]:
        """
        프로젝트를 쇼츠 목록과 함께 조회

        selectinload: 별도의 SELECT로 shorts를 미리 로드
        이렇게 하지 않으면 project.shorts 접근 시 추가 쿼리가 발생하는데
        비동기 세션에서는 이 지연 로딩(lazy loading)이 지원되지 않아 에러가 발생

        실행되는 SQL (2개)
            1. SELECT * FROM project WHERE id = ?
            2. SELECT * FROM shorts WHERE project_id IN (?)
        """

        result = await self.db.execute(
            select(Project)
            .where(Project.id == project_id)
            .options(selectinload(Project.shorts))
        )

        return result.scalar_one_or_none()

    async def get_by_status(self, status: ProjectStatus) -> list[Project]:
        """
        특정 상태의 프로젝트 목록 조회

        사용 예: 실패한 프로젝트 목록 재처리, 진행 중인 작업 모니터링
        """
        
        result = await self.db.execute(
            select(Project)
            .where(Project.status == status)
            .order_by(Project.created_at.desc())
        )

        return list(result.scalars().all())
    
    async def update_status(self, project_id: str, status: ProjectStatus, error_message: str | None = None, ) -> Optional[Project]:
        """
        프로젝트 상태 업데이트 헬퍼

        파이프라인의 각 단계에서 상태를 전환할 때 사용
        ex> PENDING -> DOWNLOADING -> ANALYZING -> DOMPLETED

        error_message는 FAILED 상태로 전환 시에만 전달
        """

        project = await self.get_by_id(project_id)      # 부모의 get_by_id 사용

        if not project:
            return None
        
        data = {"status": status}

        if error_message is not None:
            data["error_message"] = error_message

        return await self.update(project, data)         # 부모의 update 사용