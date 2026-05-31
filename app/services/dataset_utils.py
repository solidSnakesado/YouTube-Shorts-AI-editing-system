# 계층: 비즈니스 로직 계층 (Service 헬퍼) 
# 역할: dataset_builder.py 유틸 함수 분리 (300줄 원칙 대응)
# 27일차 신규: Firefox 쿠키 자동 갱신 + 기처리 영상 스킵
# 의존: sqlite3, pathlib, loguru

"""데이터셋 빌더 유틸 - 쿠키 갱신 + 기처리 ID 추적"""

import json
import sqlite3
from pathlib import Path

from loguru import logger

def load_processed_ids(output_path: Path) -> set[str]:
    """
    출력 JSON에서 이미 처리된 video_id를 추출
    재시작 시 중복 처리 방지용

    Args:
        output_path: 데이터셋 출력 파일 경로

    Returns:
        처리 완료된 video_id set
    """

    if not output_path.is_file():
        return set()
    ids: set[str] = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                vid = json.loads(line).get("metadata", {}).get("video_id")
                if vid:
                    ids.add(vid)
            except Exception:
                pass
    return ids

def refresh_firefox_cookies(cookie_path: str = "data/youtube_cookies.txt") -> bool:
    """
    Firefox 쿠키 자동 추출 - SQLite에서 직접 읽기 (암호화 없음)
    Chrome과 달리 Firefox는 DPAPI 암호화를 사용하지 않아 WSL에서 직접 읽기 가능

    우선순위:
        /mnt/c/Users/*/AppData/Roaming/Mozilla/Firefox/Profiles/*/cookies.sqlite

    Args:
        cookie_path: 출력 쿠키 파일 경로 (Netscape 포맷)

    Returns:
        True(성공) / False(Firefox 없음 또는 오류)
    """

    import glob
    import shutil
    import tempfile

    dbs = glob.glob("/mnt/c/Users/*/AppData/Roaming/Mozilla/Firefox/Profiles/*/cookies.sqlite")
    if not dbs:
        logger.warning("Firefox 쿠키 DB 없음 - Firefox 설치 및 YouTube 로그인 필요")
        return False
    
    try:
        fd, tmp = tempfile.mkstemp(suffix=".db")
        import os
        os.close(fd)
        shutil.copy2(dbs[0], tmp)
        conn = sqlite3.connect(tmp)
        rows = conn.execute("SELECT name, value, host FROM moz_cookies WHERE host LIKE '%youtube%'").fetchall()
        conn.close()
        Path(tmp).unlink(missing_ok=True)

        if not rows:
            logger.warning("Firefox에 YouTube 쿠키 없음 - YouTube 로그인 확인")
            return False
        
        with open(cookie_path, "w") as f:
            f.write("# Netscape HTTP Cookie File\n")
            for name, value, host in rows:
                f.write(f"{host}\tTRUE\t/\tFALSE\t0\t{name}\t{value}\n")

        logger.debug(f"Firefox 쿠키 갱신 완료: {len(rows)}개")
        return True
    except Exception as e:
        logger.warning(f"Firefox 쿠키 추출 실패: {e}")
        return False