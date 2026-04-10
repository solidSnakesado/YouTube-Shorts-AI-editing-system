# 계층: 데이터 접근 계층 (Repository)
# 역할: Shorts 엔티이에 특화된 DB 조회 메서드를 제공
# 의존: BaseRepository, Shorts 모델
# MVA 원칙: 레포지토리 패턴 - SQL 쿼리는 이 계층에만 존재

"""
쇼츠 클립 레포리토리
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import Shorts, ShortStatus
from app.repositories.base_repository import BaseRepository

class ShortsRepository(BaseRepository[Shorts]):
    """
    쇼츠 전용 레포지토리

    상속받는 메서드 (BaseRepository):
        create, get_by_id, get_all, count, update, delete

    추가 메서드:
        get_by_project()            -> 특정 프로젝트의 쇼츠 전체 (시간순)
        get_completed_by_project()  -> 완료된 쇼츠만 (점수 높은 순)
    """

    def __init__(self, db: AsyncSession):
        super().__init__(Shorts, db)

    async def get_by_project(self, project_id: str) -> list[Shorts]:
        """
        특정 프로젝트의 쇼츠 전체 목록을 시작 시점 순으로 조회

        API에서 프로젝트 상세 페이지에 쇼츠 목록을 표시할 때 사용.
        start_sec 오름차순 -> 영상의 앞부분 하이라이트가 먼저 나옴
        """
        
        result = await self.db.execute(
            select(Shorts)
            .where(Shorts.project_id == project_id)
            .order_by(Shorts.start_sec)             # 영상 타임라인 순서대로
        )

        return list(result.scalars().all())
    
    async def get_completed_by_project(self, project_id: str) -> list[Shorts]:
        """
        특정 프로젝트에서 편집 완료된 쇼츠만 조회 (흥미도 높은 순)

        hook_score 내림차순 -> 가장 매력적인 쇼츠가 먼저 나옴.
        다운로드 페이지나 유투브 업로드 목록에서 사용
        """
        
        result = await self.db.execute(
            select(Shorts)
            .where(
                Shorts.project_id == project_id,
                Shorts.status == ShortStatus.COMPLETED,     # 완료된 것만 필터
            )
            .order_by(Shorts.hook_score.desc())             # 점수 높은 순
        )

        return list(result.scalars().all())