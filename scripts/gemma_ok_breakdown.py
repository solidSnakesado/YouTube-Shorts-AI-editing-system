#!/usr/bin/env python3
"""
52일차: OK-rate 점수대 x 탐색 x 계층태그 분해 집계

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_ok_breakdown.py
- 수정 이력: 1차 작성 (52일차)

[배경]
- 51일차 교훈 3: 헤드라인 OK율만으로 판정 금지 — 점수대 분해에서 계층화(단조성)
  같은 실질 변화가 드러남. 이를 측정 표준 절차로 스크립트화.
- 52일차 추가: 계층 발행 태깅([고신뢰]/[보충]/[탐색] reason 프리픽스) 분해 포함.

[집계 기준]
- usable = OK + 선택NO (경계/편집NO 제외 — measure_ok_rate.py와 동일)
- feedback은 enum 대문자(OK/NO) 저장 (36일차 확정)

[실행]
  python3 scripts/gemma_ok_breakdown.py --model-filter round15_fb2
"""

import argparse
import math
import sqlite3
import sys
from collections import defaultdict

DB_PATH = "data/shorts_ai.db"


def ci_half(p: float, n: int) -> float:
    """95% CI 반폭 (정규근사) — measure_ok_rate.py와 동일식."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(p * (1.0 - p) / n)


def band(score: float) -> str:
    """hook_score -> 점수대 라벨 (1.0 이상은 과신 구간으로 별도)."""
    if score >= 1.0:
        return "1.0+"
    return f"0.{int(score * 10)}대"


def tier_tag(reason: str) -> str:
    """52일차: highlight_reason 프리픽스([고신뢰] 등) 추출. 없으면 무태그."""
    r = reason or ""
    if r.startswith("[") and "]" in r:
        return r[1: r.index("]")]
    return "무태그"


def print_group(title: str, data: dict) -> None:
    print(f"--- {title}")
    for key in sorted(data):
        ok, n = data[key]
        p = ok / n if n else 0.0
        print(f"  {key:<8}: {ok:>3}/{n:<3} = {p * 100:5.1f}% (CI ±{ci_half(p, n) * 100:.1f}%p)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-filter", default="round15_fb2",
                    help="model_version LIKE 부분 문자열")
    args = ap.parse_args()

    try:
        con = sqlite3.connect(DB_PATH)
    except Exception as e:                          # noqa: BLE001
        print(f"DB 연결 실패 ({DB_PATH}): {e}")
        return 1

    # usable(OK + 선택NO)만 — 알 수 없는 NO 사유는 선택NO 취급(measure와 동일)
    rows = con.execute(
        "SELECT hook_score, feedback, is_exploration, highlight_reason "
        "FROM shorts WHERE model_version LIKE ? AND feedback IS NOT NULL "
        "AND (UPPER(feedback)='OK' OR (UPPER(feedback)='NO' "
        "     AND (feedback_reason IS NULL OR feedback_reason='selection')))",
        (f"%{args.model_filter}%",),
    ).fetchall()
    con.close()
    if not rows:
        print(f"해당 라벨 없음 (filter={args.model_filter})")
        return 1

    by_band = defaultdict(lambda: [0, 0])
    by_expl = defaultdict(lambda: [0, 0])
    by_tier = defaultdict(lambda: [0, 0])
    by_band_exploit = defaultdict(lambda: [0, 0])   # 활용 픽만의 점수대 (51일차 비교 축)

    for score, fb, expl, reason in rows:
        ok = 1 if (fb or "").upper() == "OK" else 0
        b = band(float(score or 0.0))
        is_expl = bool(expl)
        for agg, key in ((by_band, b),
                         (by_expl, "탐색" if is_expl else "활용"),
                         (by_tier, tier_tag(reason))):
            agg[key][0] += ok
            agg[key][1] += 1
        if not is_expl:
            by_band_exploit[b][0] += ok
            by_band_exploit[b][1] += 1

    total_ok = sum(1 for _s, fb, _e, _r in rows if (fb or "").upper() == "OK")
    n = len(rows)
    print("=" * 60)
    print(f"분해 집계 (filter={args.model_filter}) | usable {n} | "
          f"전체 OK율 {total_ok / n * 100:.1f}%")
    print_group("점수대 (전체)", by_band)
    print_group("점수대 (활용 픽만)", by_band_exploit)
    print_group("활용/탐색", by_expl)
    print_group("계층태그", by_tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())