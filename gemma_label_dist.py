# 44일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_label_dist.py
#
# 목적: round4 진동(step100=0.9 / step200=0.111 / step300=랜덤) 원인이
#   "라벨 특정값 군집"인지 정밀 확인. 0.1 버킷 히스토그램은 군집을 못 봄
#   -> 0.01 버킷 + 최빈값 top + distinct 개수 + 군집률로 판단.
#
#   판정:
#     - 소수 값에 군집(top10 값이 50%+, distinct 적음) -> 지터 필요(Qwen 정신:
#       연속분포 보존). heatmap 평균이 특정 소수값에 모이는 구조.
#     - 매끄러운 연속(distinct 많음, 최빈값도 낮은 비율) -> 라벨은 정상,
#       모델 이산출력은 학습 설정(epoch/lr) 문제 -> 지터 무의미.
#
# 의존(번들): gemma_collate.py(parse_sample), gemma_collapse_check.py(parse_hook_score).
# 실행(로컬 WSL, 재라벨 데이터 대상):
#   python gemma_label_dist.py --train datasets/gemma_audio/train_relabel.jsonl
from __future__ import annotations

import argparse
import json
from collections import Counter

from gemma_collapse_check import parse_hook_score


def collect_labels(path: str):
    """pos/neg 각각의 hook_score 라벨 수집(label 기반 분리)."""
    from gemma_collate import parse_sample

    pos: list[float] = []
    neg: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            is_neg = row.get("metadata", {}).get("label") == "negative"
            try:
                v = parse_hook_score(parse_sample(row)["target"])
            except Exception:                       # noqa: BLE001
                continue
            if v < 0:
                continue
            (neg if is_neg else pos).append(round(v, 4))
    return pos, neg


def analyze(name: str, scores: list[float]) -> None:
    """0.01 버킷 군집 + 최빈값 top10 + distinct + 군집률."""
    print(f"\n{'='*60}\n{name} (n={len(scores)})")
    if not scores:
        print("없음")
        return
    n = len(scores)
    distinct = len(set(scores))
    print(f"distinct 값 개수: {distinct}  (전체 대비 {distinct/n:.1%})")

    # 0.01 버킷 분포
    b001 = Counter(round(s, 2) for s in scores)
    nonzero = sum(1 for c in b001.values() if c > 0)
    print(f"0.01 버킷 사용 개수: {nonzero}/101")

    # 최빈값 top10
    cnt = Counter(scores)
    top = cnt.most_common(10)
    top_sum = sum(c for _, c in top)
    print(f"최빈값 top10이 전체의 {top_sum/n:.1%} 차지:")
    for v, c in top:
        bar = "#" * int(40 * c / top[0][1])
        print(f"   {v:.4f} : {c:5d} ({c/n:5.1%}) {bar}")

    # 군집 판정
    top10_ratio = top_sum / n
    verdict = ("군집 심함 -> 지터 필요" if top10_ratio > 0.4 or distinct / n < 0.3
               else "매끄러운 연속 -> 지터 무의미(학습설정 의심)")
    print(f"판정: {verdict}  (top10={top10_ratio:.1%}, distinct비율={distinct/n:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="라벨 분포 정밀 진단(군집 여부)")
    ap.add_argument("--train", default="datasets/gemma_audio/train_relabel.jsonl")
    args = ap.parse_args()

    pos, neg = collect_labels(args.train)
    analyze("POS 라벨", pos)
    analyze("NEG 라벨", neg)
    analyze("전체(pos+neg)", pos + neg)


if __name__ == "__main__":
    main()