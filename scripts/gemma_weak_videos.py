#!/usr/bin/env python3
"""
51일차: Gemma round12 하위 성적 영상 육안 확인용 조회 스크립트

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_weak_videos.py
- 수정 이력: 1차 작성 (51일차)

[용도]
- 50일차 OK-rate 분석에서 OK율 하위였던 영상 5개의
  제목 / URL / 길이 / 쇼츠별 결과(hook_score, 피드백, 탐색 여부)를 출력
- 사용자가 URL을 열어 장르/유형을 육안 확인하는 데 필요한 정보만 제공

[실행]
  python3 scripts/gemma_weak_videos.py
  python3 scripts/gemma_weak_videos.py --db data/shorts_ai.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# 51일차: 50일차 핸드오프 1-4절 기재 하위 영상 5개 (프로젝트 ID 접두사)
WEAK_PREFIXES = [
    "7bbd2f84",   # 2/7
    "2955250d",   # 2/5
    "3049ae0d",   # 3/7
    "a7db77f2",   # 1/2
    "d511149f",   # 3/5
]


def fmt_dur(sec):
    # 51일차: 초 → mm:ss 표기
    if sec is None:
        return "?"
    sec = int(sec)
    return f"{sec // 60}분 {sec % 60:02d}초"


def main():
    parser = argparse.ArgumentParser(description="Gemma round12 하위 영상 조회")
    parser.add_argument("--db", default="data/shorts_ai.db", help="SQLite DB 경로")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[오류] DB 파일 없음: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for prefix in WEAK_PREFIXES:
        # 51일차: 접두사로 프로젝트 조회
        cur.execute(
            "SELECT id, title, youtube_url, duration_sec FROM projects WHERE id LIKE ?",
            (prefix + "%",),
        )
        rows = cur.fetchall()
        if not rows:
            print(f"\n=== {prefix}* : 프로젝트 없음 ===")
            continue
        if len(rows) > 1:
            print(f"\n[경고] {prefix}* 접두사에 프로젝트 {len(rows)}건 매칭 — 전부 출력")

        for proj in rows:
            print("\n" + "=" * 70)
            print(f"프로젝트: {proj['id']}")
            print(f"제목    : {proj['title']}")
            print(f"URL     : {proj['youtube_url']}")
            print(f"길이    : {fmt_dur(proj['duration_sec'])}")

            # 51일차: round12 쇼츠만 (부분 문자열 매칭 — measure_ok_rate.py와 동일 방식)
            cur.execute(
                """
                SELECT start_sec, end_sec, hook_score, feedback, feedback_reason,
                       is_exploration, title_suggestion
                FROM shorts
                WHERE project_id = ? AND model_version LIKE '%round12%'
                ORDER BY start_sec
                """,
                (proj["id"],),
            )
            shorts = cur.fetchall()
            ok = sum(1 for s in shorts if s["feedback"] == "ok")
            labeled = sum(1 for s in shorts if s["feedback"] is not None)
            print(f"쇼츠    : {len(shorts)}건 (라벨 {labeled}, OK {ok})")
            print("-" * 70)
            for s in shorts:
                fb = s["feedback"] or "미평가"
                reason = f"/{s['feedback_reason']}" if s["feedback_reason"] else ""
                exp = " [탐색]" if s["is_exploration"] else ""
                hs = f"{s['hook_score']:.3f}" if s["hook_score"] is not None else "  ?  "
                start = int(s["start_sec"])
                end = int(s["end_sec"])
                print(
                    f"  {start // 60:3d}:{start % 60:02d}~{end // 60:3d}:{end % 60:02d}"
                    f"  score={hs}  {fb}{reason}{exp}  | {s['title_suggestion'] or ''}"
                )

    conn.close()


if __name__ == "__main__":
    main()