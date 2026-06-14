# 계층: 스크립트 (CLI 진입점)
# 역할: 기존 shorts_ai.db의 shorts 테이블에 33일차 피드백 칼럼 추가 (일회성 마이그레이션)
# 의존: app.core.config(settings.DATABASE_URL) - DB 경로 하드코딩 금지
# 33일차 신규: init_db()의 create_all은 신규 테이블만 생성 -> 기존 테이블 칼럼 추가는 ALTER 필요
#   모든 신규 칼럼은 nullable이라 기존 행에 안전 (SQLite ADD COLUMN). 재실행 가능(idempotent).

"""피드백 컬럼 마이그레이션 - domain.py Short 모델 변경분을 기존 DB에 반영"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.core.config import settings

# domain.py Short에 추가된 컬럼 -> SQLite 타입
# datetime/enum=TEXT, bool=INTEGER (SQLite의 SQLite 매핑과 일차)
NEW_COLUMNS = {
    "feedback": "TEXT",
    "feedback_reason": "TEXT",
    "feedback_at": "TEXT",
    "is_exploration": "INTEGER",
    "train_sample_json": "TEXT",
    "model_version": "TEXT",
}

def _db_path_from_url(url: str)-> str:
    """settings.DATABASE_URL에서 SQLite 파일 경로 추출 (하드코딩 금지)"""

    # ex> "sqlite+aiosqlite:///./data/shorts_ai.db" -> "./data/shorts_ai.db"
    prefix = url.split(":///")
    if len(prefix) != 2:
        raise SystemExit(f"SQLite URL 형식이 아님: {url}")
    return prefix[1]

def migrate(db_path: str) -> None:
    """shorts 테이블에 누락된 피드백 컬럼만 추가"""
    
    if not Path(db_path).is_file():
        raise SystemExit(f"DB 파일 없음: {db_path}")
    
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        existing = {row[1] for row in cur.execute("PRAGMA table_info(shorts)")}
        if not existing:
            raise SystemExit("shorts 테이블 없음 (먼저 서버 1회 기동으로 init_db 필요)")
        
        added = []
        for col, col_type in NEW_COLUMNS.items():
            if col in existing:
                logger.info(f"이미 존재: {col} (건너뜀)")
                continue
            cur.execute(f"ALTER TABLE shorts ADD COLUMN {col} {col_type}")
            added.append(col)
            logger.info(f"추가: {col} {col_type}")

        con.commit()
        logger.info(f"마이그레이션 완료 | 추가 {len(added)}개: {added or '없음'}")
    finally:
        con.close()

def main() -> None:
    db_path = _db_path_from_url(settings.DATABASE_URL)
    logger.info(f"대상 DB: {db_path}")
    migrate(db_path)

if __name__ == "__main__":
    main()