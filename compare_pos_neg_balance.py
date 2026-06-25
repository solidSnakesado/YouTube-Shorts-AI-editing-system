#!/usr/bin/env python3
# 계층: 검증 유틸 (루트 실행 스크립트)
# 역할: 포지티브/네거티브 데이터셋의 영상 단위 1:1 매칭 검증 (39일차 네거티브 빌드 직후)
# 39일차 신규: dataset.jsonl(포지티브) vs dataset_neg.jsonl(네거티브)을 video_id별로 대조
#   - 각 샘플을 assistant 출력으로 pos/neg 자동 분류 (병합본도 안전)
#   - 영상별 pos 수 vs neg 수 비교 -> 일치/미달/초과/누락/고아 분류
#   - 초과(neg>pos)·고아(pos 없는 neg)는 빌더 버그 신호 -> 명시 경고
#   미디어는 보지 않음(개수만). 스키마/파일 실존은 verify_gemma_dataset.py 담당

"""포지티브/네거티브 영상 단위 1:1 매칭 검증 CLI."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_POS = "datasets/gemma_audio/dataset.jsonl"
DEFAULT_NEG = "datasets/gemma_audio/dataset_neg.jsonl"


def load_jsonl(path: Path):
    """jsonl 로드 -> (행 리스트, 파싱실패 행번호 리스트)."""

    rows, errors = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(i)
    return rows, errors


def classify(sample: dict) -> str:
    """assistant 출력 파싱 -> 'pos' | 'neg' | 'bad'. (verify와 동일 규칙)"""

    try:
        text = sample["messages"][1]["content"][0]["text"]
        highlights = json.loads(text).get("highlights", None)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "bad"
    if highlights == []:
        return "neg"
    if isinstance(highlights, list) and len(highlights) == 1 and "hook_score" in highlights[0]:
        return "pos"
    return "bad"


def tally(path: Path):
    """파일 내 샘플을 분류해 (pos_counter, neg_counter, bad수, 파싱실패수) 반환."""

    rows, parse_errs = load_jsonl(path)
    pos, neg = Counter(), Counter()
    bad = 0
    for sample in rows:
        vid = sample.get("metadata", {}).get("video_id")
        kind = classify(sample)
        if kind == "pos" and vid:
            pos[vid] += 1
        elif kind == "neg" and vid:
            neg[vid] += 1
        else:
            bad += 1
    return pos, neg, bad, len(parse_errs)


def _parse_args():
    ap = argparse.ArgumentParser(description="포지티브/네거티브 1:1 매칭 검증")
    ap.add_argument("--pos-path", default=DEFAULT_POS, help="포지티브 jsonl")
    ap.add_argument("--neg-path", default=DEFAULT_NEG, help="네거티브 jsonl")
    ap.add_argument("--show", type=int, default=10, help="불일치 영상 표시 개수")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    pos_path, neg_path = Path(args.pos_path), Path(args.neg_path)
    for p in (pos_path, neg_path):
        if not p.exists():
            print(f"파일 없음: {p}")
            return 1

    # 양쪽 파일을 분류 집계 후 합산(병합본/혼합 파일도 안전)
    pos_a, neg_a, bad_a, perr_a = tally(pos_path)
    pos_b, neg_b, bad_b, perr_b = tally(neg_path)
    pos_counts = pos_a + pos_b
    neg_counts = neg_a + neg_b

    pos_total = sum(pos_counts.values())
    neg_total = sum(neg_counts.values())
    print("=== 포지티브/네거티브 1:1 매칭 검증 ===")
    print(f"포지티브 파일: {pos_path}")
    print(f"네거티브 파일: {neg_path}")
    print(f"\n[총계] 포지티브 {pos_total}개 / 네거티브 {neg_total}개 "
          f"(차이 {pos_total - neg_total:+d})")
    print(f"[영상수] 포지티브 보유 {len(pos_counts)} / 네거티브 보유 {len(neg_counts)}")
    if bad_a or bad_b or perr_a or perr_b:
        print(f"[주의] 불량 샘플 pos{bad_a}/neg{bad_b}, 파싱실패 pos{perr_a}/neg{perr_b}")

    # 영상 단위 비교
    matched = under = over = 0
    missing_neg, orphan_neg, under_list, over_list = [], [], [], []
    for vid in set(pos_counts) | set(neg_counts):
        p, n = pos_counts.get(vid, 0), neg_counts.get(vid, 0)
        if p > 0 and n == 0:
            missing_neg.append((vid, p))
        elif n > 0 and p == 0:
            orphan_neg.append((vid, n))
        elif n == p:
            matched += 1
        elif n < p:
            under += 1
            under_list.append((vid, p, n))
        else:
            over += 1
            over_list.append((vid, p, n))

    print("\n[영상별 매칭]")
    print(f"  일치(neg==pos)   : {matched}")
    print(f"  미달(neg<pos)    : {under}  (네거티브 부족)")
    print(f"  초과(neg>pos)    : {over}  (1:1 위반 — 버그 신호)")
    print(f"  네거티브 누락     : {len(missing_neg)}  (pos 있는데 neg 0)")
    print(f"  고아 네거티브     : {len(orphan_neg)}  (pos 없는데 neg 존재 — 버그 신호)")

    if under_list:
        deficit = sum(p - n for _, p, n in under_list)
        print(f"\n  미달 영상 (총 부족 {deficit}개, 상위 {args.show}):")
        for vid, p, n in sorted(under_list, key=lambda x: x[1] - x[2], reverse=True)[:args.show]:
            print(f"    {vid}: pos={p} neg={n} (부족 {p - n})")
    if over_list:
        print(f"\n  ⚠️ 초과 영상 (상위 {args.show}):")
        for vid, p, n in sorted(over_list, key=lambda x: x[2] - x[1], reverse=True)[:args.show]:
            print(f"    {vid}: pos={p} neg={n} (초과 {n - p})")
    if missing_neg:
        print(f"\n  네거티브 누락 영상 (상위 {args.show}):")
        for vid, p in missing_neg[:args.show]:
            print(f"    {vid}: pos={p} neg=0")
    if orphan_neg:
        print(f"\n  ⚠️ 고아 네거티브 영상 (상위 {args.show}):")
        for vid, n in orphan_neg[:args.show]:
            print(f"    {vid}: pos=0 neg={n}")

    # 종합: 완전 1:1이면 통과. 미달은 허용 가능(오디오 스킵 등), 초과/고아는 버그
    perfect = (under == 0 and over == 0 and not missing_neg and not orphan_neg)
    has_bug = (over > 0 or len(orphan_neg) > 0)
    if perfect:
        verdict = "완전 1:1 일치"
    elif has_bug:
        verdict = "버그 의심 (초과/고아 발견 — 빌더 점검 필요)"
    else:
        verdict = "경미한 미달만 존재 (neg<pos, 오디오 스킵 등 — 수용 여부 판단)"
    print(f"\n=== 종합: {verdict} ===")
    return 0 if perfect else 1


if __name__ == "__main__":
    sys.exit(main())