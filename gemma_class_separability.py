# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_class_separability.py
#
# 목적: 단일점수 회귀 5번 붕괴 후 분류 버킷 전환(방향1) 전, 학습 없이
#   "점수를 등급으로 나눴을 때 pos/neg가 깔끔히 분리되는가"를 데이터만으로 확인.
#   pos 0.236~1.0 / neg 0.0~0.5로 점수가 겹치므로(0.236~0.5 공존), 등급 경계가
#   이 겹침을 어떻게 가르는지가 분류 성공의 선결 조건.
#   분리 안 되면(상위등급에 neg 다수, 하위등급에 pos 다수) 분류도 헛수고 -> 다른 방향.
#
# 측정:
#   1) pos/neg 기본 분포(n/min/max/mean/std)
#   2) 겹침 정도(pos 중 임계미만 비율, neg 중 임계이상 비율)
#   3) 2클래스 임계 스윕(0.30~0.70): 임계별 다수결 정확도 상한 + 혼입
#   4) 3/5클래스 등급별 pos/neg 분포 + 등급 순도(다수결 정확도 상한)
#
# 판정 기준(참고): 2클래스 최적 임계에서 정확도 상한 70%+ & 혼입 양방향 30%- 이면
#   분류 분리 가능. 60% 미만이면 점수 자체에 클래스 신호 약함 -> 분류도 위험(다른 방향).
#
# 의존(번들): gemma_collate.py(parse_sample), gemma_collapse_check.py(parse_hook_score).
# 실행(로컬 WSL):
#   python gemma_class_separability.py --train datasets/gemma_audio/train_relabel.jsonl
from __future__ import annotations

import argparse
import json
import statistics as st

from gemma_collapse_check import parse_hook_score


def collect_labels(path: str):
    """pos/neg 각각의 hook_score 라벨 수집(metadata.label 기반 분리)."""
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
            (neg if is_neg else pos).append(v)
    return pos, neg


def describe(name: str, scores: list[float]) -> None:
    """기본 분포(n/min/max/mean/std + 0.8 이상 비율)."""
    if not scores:
        print(f"{name}: 없음")
        return
    mean = sum(scores) / len(scores)
    std = st.pstdev(scores) if len(scores) > 1 else 0.0
    p80 = sum(1 for s in scores if s >= 0.8) / len(scores)
    print(f"{name}: n={len(scores)} min={min(scores):.3f} max={max(scores):.3f} "
          f"mean={mean:.3f} std={std:.3f} (0.8+ {p80:.1%})")


def overlap_report(pos: list[float], neg: list[float], t: float) -> None:
    """겹침 정도: 하위로 새는 pos, 상위로 새는 neg."""
    print(f"\n{'=' * 60}\n겹침 분석 (기준 임계 {t})")
    pos_low = sum(1 for s in pos if s < t) / len(pos) if pos else 0
    neg_high = sum(1 for s in neg if s >= t) / len(neg) if neg else 0
    print(f"  pos 중 {t} 미만 (하위로 새는 pos): {pos_low:.1%}")
    print(f"  neg 중 {t} 이상 (상위로 새는 neg): {neg_high:.1%}")
    print(f"  -> 양방향 혼입 합 {pos_low + neg_high:.1%} (낮을수록 깔끔)")


def threshold_sweep(pos: list[float], neg: list[float]) -> None:
    """2클래스 임계 스윕: 임계별 다수결 정확도 상한 + 혼입."""
    print(f"\n{'=' * 60}\n2클래스 임계 스윕 (상=pos / 하=neg 가정)")
    print(f"{'임계':>6} {'정확도상한':>10} {'pos→상':>8} {'neg→하':>8} {'혼입합':>8}")
    n_all = len(pos) + len(neg)
    best = (0.0, 0.0)
    for i in range(30, 71, 5):
        t = i / 100
        pos_hi = sum(1 for s in pos if s >= t)
        neg_lo = sum(1 for s in neg if s < t)
        acc = (pos_hi + neg_lo) / n_all if n_all else 0
        pos_rate = pos_hi / len(pos) if pos else 0
        neg_rate = neg_lo / len(neg) if neg else 0
        mix = (1 - pos_rate) + (1 - neg_rate)
        if acc > best[1]:
            best = (t, acc)
        print(f"{t:>6.2f} {acc:>10.1%} {pos_rate:>8.1%} {neg_rate:>8.1%} {mix:>8.1%}")
    print(f"  최적 임계 {best[0]:.2f} → 정확도 상한 {best[1]:.1%}")


def multiclass_report(pos: list[float], neg: list[float],
                      edges: list[float], names: list[str]) -> None:
    """N클래스 등급별 pos/neg 분포 + 순도(등급 다수결 정확도 상한)."""
    print(f"\n{'=' * 60}\n{len(names)}클래스 등급 분포 (경계 {edges})")

    def bucket(v: float) -> int:
        for k, e in enumerate(edges):
            if v < e:
                return k
        return len(edges)

    grid = [[0, 0] for _ in names]              # [pos, neg]
    for v in pos:
        grid[bucket(v)][0] += 1
    for v in neg:
        grid[bucket(v)][1] += 1

    n_all = len(pos) + len(neg)
    major_sum = 0
    print(f"{'등급':>6} {'pos':>6} {'neg':>6} {'순도':>8} {'다수':>6}")
    for nm, (p, g) in zip(names, grid):
        tot = p + g
        if tot == 0:
            print(f"{nm:>6} {p:>6} {g:>6} {'-':>8} {'-':>6}")
            continue
        major = max(p, g)
        major_sum += major
        who = "pos" if p >= g else "neg"
        print(f"{nm:>6} {p:>6} {g:>6} {major / tot:>8.1%} {who:>6}")
    acc = major_sum / n_all if n_all else 0
    print(f"  등급 다수결 정확도 상한: {acc:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="분류 버킷 분리도 진단(학습0)")
    ap.add_argument("--train",
                    default="datasets/gemma_audio/train_relabel.jsonl")
    args = ap.parse_args()

    pos, neg = collect_labels(args.train)
    print(f"{'=' * 60}\n기본 분포")
    describe("POS", pos)
    describe("NEG", neg)

    overlap_report(pos, neg, 0.5)
    threshold_sweep(pos, neg)
    multiclass_report(pos, neg, [0.33, 0.66], ["하", "중", "상"])
    multiclass_report(pos, neg, [0.2, 0.4, 0.6, 0.8], ["1", "2", "3", "4", "5"])


if __name__ == "__main__":
    main()