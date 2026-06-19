# 36일차: round2 OK율 측정 스크립트
# OK율 = OK / (OK + 선택NO) - 경계NO / 편집NO 는 분모에서 제외
# 사용:
#   python3 measure_ok_rate.py          # 모든 model_version 출력
#   python3 measure_ok_rate.py round2   # 특정 버전만 (부분 문자열 매칭)

import sqlite3
import sys
import math

DB_PATH = "data/shorts_ai.db"
# 비교 기준 (핸드오프 2-3): 베이스 41.0% -> round1b 50.3% -> round2 ?
BASELINES = {"base": 41.0, "round1b": 50.3}
ROUND1B = 50.3  # 핵심 비교 대상

def fetch_counts(con):
    """model_version 별 OK / 선택NO / 경계NO / 편집NO 집계"""
    
    rows = con.execute(
        "SELECT model_version, feedback, feedback_reason, COUNT(*) "
        "FROM shorts WHERE feedback IS NOT NULL "
        "GROUP BY model_version, feedback, feedback_reason"
    ).fetchall()
    data = {}
    for mv, fb, reason, cnt in rows:
        mv = mv if mv else "(none)"
        d = data.setdefault(
            mv, {"ok": 0, "no_selection": 0, "no_boundary": 0, "no_editing": 0}
        )
        fb_u = (fb or "").upper()       # 36일차: DB는 enum 이름(대문자 OK/NO)으로 저장 -> 대문자 매칭
        if fb_u == "OK":
            d["ok"] += cnt      # OK는 사유 무관 전부 집계
        elif fb_u == "NO":
            key = "no_" + (reason or "selection").lower()
            if key in d:
                d[key] += cnt
            else:
                d["no_selection"] += cnt    # 알 수 없는 사유는 선택NO로 처리
    return data

def ci_half_width(p, n):
    """95% 신뢰구간 반폭 (정규근사). n=0이면 None."""

    if n <= 0:
        return None
    return 1.96 * math.sqrt(p * (1.0 - p) / n)

def report(mv, d):
    ok = d["ok"]
    sel = d["no_selection"]
    usable = ok + sel
    print(f"\n=== {mv} ===")
    print(
        f"  OK: {ok} | 선택NO: {sel} | "
        f"경계NO: {d['no_boundary']} | 편집NO: {d['no_editing']}"
    )
    if usable == 0:
        print(" OK율: 측정 불가 (OK + 선택NO = 0)")
        return
    rate = ok / usable * 100.0
    half = ci_half_width(ok / usable, usable) * 100.0
    print(f"    usable(OK+선택NO): {usable}")
    print(f"    >> OK율 = {ok}/{usable} = {rate:.1f}% (95% CI ±{half:.1f}%p)")
    for name, base in BASELINES.items():
        print(f"    vs {name} {base}%: {rate - base:+.1f}%p")
    lo, hi = rate - half, rate + half
    if lo > ROUND1B:
        print(f"        -> round1b({ROUND1B}%)보다 유의하게 높음 (CI 하한 {lo:.1f}% > {ROUND1B}%)")
    elif hi < ROUND1B:
        print(f"        -> round1b({ROUND1B}%)보다 유의하게 낮음 (CI 상한 {hi:.1f}% < {ROUND1B}%)")
    else:
        print(
            f"      -> round1b({ROUND1B}%)와 구별 불가 "
            f"(CI {lo:.1f}~{hi:.1f}% 가 {ROUND1B}% 포함) - 라벨 더 필요"
        )

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        con = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        print(f"DB 연결 실패 ({DB_PATH}): {e}")
        return
    try:
        data = fetch_counts(con)
    finally:
        con.close()

    if not data:
        print("라벨된 쇼츠가 없습니다 (feedback IS NULL)")
        return
    
    if target:
        matched = {k: v for k, v in data.items() if target in k}
        if not matched:
            print(f"model_version에 '{target}' 포함된 항목 없음")
            print(f"존재하는 버전: {sorted(data.keys())}")
            return
        for mv in sorted(matched.keys()):
            report(mv, matched[mv])
    else:
        for mv in sorted(data.keys()):
            report(mv, data[mv])

if __name__ == "__main__":
    main()