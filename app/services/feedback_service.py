# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 쇼츠에 대한 사람 OK/NO 피드백 저장 (피드백 루프 C 구성요소)
# 의존: ShortRepository (DI로 주입 받음)
# MVA 원칙: 서비스 = 순수 비즈니스 로직, SQL은 레포지토리에 위임
# 33일차 신규: analysis_service.py가 299줄이라 300줄 규칙에 따라 별도 모듈로 분리
#   피드백 관련 로직(추후 D:학습데이터 변환 등)은 이 모듈에서 확장

"""피드백 서비스 - 사람 OK/NO 평가 저장"""

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.models.domain import FeedbackLabel, Shorts
from app.repositories.shorts_repository import ShortsRepository

class FeedbackService:
    """피드백 서비스 - 쇼츠에 대한 사람 평가(OK/NO) 기록"""

    def __init__(self, shorts_repo: ShortsRepository):
        """DI로 레포지토리 주입"""

        self.shorts_repo = shorts_repo

    async def submit_feedback(
        self, shorts_id: str, feedback: str, reason: Optional[str] = None
    ) -> Optional[Shorts]:
        """쇼츠에 사람 피드백 저장

        - feedback: "ok" | "no" (스키마 Literal 검증 후 전달됨)
        - reason: No 사유 (selection/boundary/editing). OK일 때는 무시(None 저장)
          -> 입도 분리: 경계 문제로 NO인데 선택 문제로 학습되는 오염 방지
        - feedback_at: 저장 시각 (재학습 배치/라운드별 지표용)
        """

        shorts = await self.shorts_repo.get_by_id(shorts_id)
        if not shorts:
            logger.error(f"피드백 대상 쇼츠 없음: {shorts_id}")
            return None
        
        label = FeedbackLabel(feedback)
        fields = {
            "feedback": label,
            "feedback_reason": reason if label == FeedbackLabel.NO else None,
            "feedback_at": datetime.now(timezone.utc),
        }
        updated = await self.shorts_repo.update(shorts, fields)
        logger.info(
            f"피드백 저장: {shorts_id} | {label.value}"
            + (f"   ({fields['feedback_reason']})" if fields["feedback_reason"] else "")
        )
        return updated