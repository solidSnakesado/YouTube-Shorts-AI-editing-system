# 44일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_data_diag.py
#
# 목적: round2가 전부 [](always-neg)로 붕괴 -> 원인 규명. knob 찍기 전에 데이터부터 본다.
#   #1 train.jsonl에 pos 타겟(실제 hook_score)이 제대로 있는가(빌드 버그면 데이터 재빌드가 답)
#   #2 pos/neg 실측 비율(불균형이면 다수클래스 붕괴)
#   #3 hook_score 분포(고정값/저분산이면 라벨 붕괴 유발 - 메모리 B-2)
#   #4 target 원문 샘플(형식 확인)
#
# GPU 불필요(jsonl만 읽음) -> 학습과 무관하게 즉시 실행.
# 의존: 표준 라이브러리만. (본 파일만 업로드)
# 실행(Colab, 추출된 상태):
#   python gemma_data_diag.py
#   python gemma_data_diag.py --jsonl datasets/gemma_audio/eval.jsonl --samples 4
from __future__ import annotations

import argparse
import json
import statistics


def target_text(row: dict):
    """row -> assistant target 텍스트(messages[1].content[0].text). 구조 이상 시 None."""
    try:
        return row["messages"][1]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None


def classify(t):
    """target 텍스트 -> ('neg', None) | ('pos', hook_score) | ('bad', None)."""
    if t is None:
        return "bad", None
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return "bad", None
    hl = obj.get("highlights")
    if not isinstance(hl, list):
        return "bad", None
    if len(hl) == 0:
        return "neg", None
    first = hl[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return "pos", float(first["hook_score"])
        except (TypeError, ValueError):
            return "bad", None
    return "bad", None


def histogram(scores: list[float]) -> None:
    """0.0~1.0을 0.1 단위 10구간으로 막대 출력."""
    buckets = [0] * 10
    for s in scores:
        idx = min(int(s * 10), 9)
        buckets[idx] += 1
    mx = max(buckets) or 1
    for i, c in enumerate(buckets):
        bar = "#" * int(40 * c / mx)
        print(f"  [{i/10:.1f}~{(i+1)/10:.1f}) {c:5d} {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description="train.jsonl 분포 진단([]-붕괴 원인)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/train.jsonl")
    ap.add_argument("--samples", type=int, default=3, help="pos/neg 원문 샘플 출력 수")
    args = ap.parse_args()

    pos_scores: list[float] = []
    n_pos = n_neg = n_bad = 0
    pos_samples: list[str] = []
    neg_samples: list[str] = []
    n_highlights: list[int] = []        # pos의 highlights 개수(보통 1)

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = target_text(row)
            kind, score = classify(t)
            if kind == "pos":
                n_pos += 1
                pos_scores.append(score)
                try:
                    n_highlights.append(len(json.loads(t)["highlights"]))
                except Exception:           # noqa: BLE001
                    pass
                if len(pos_samples) < args.samples:
                    pos_samples.append(t)
            elif kind == "neg":
                n_neg += 1
                if len(neg_samples) < args.samples:
                    neg_samples.append(t)
            else:
                n_bad += 1

    total = n_pos + n_neg + n_bad
    print(f"=== {args.jsonl} 진단 (총 {total}행) ===")
    print(f"pos(하이라이트 有): {n_pos}  neg([]): {n_neg}  bad(형식이상): {n_bad}")
    if total:
        print(f"pos 비율: {n_pos/total:.1%}  neg 비율: {n_neg/total:.1%}")

    print("\n--- #1 pos 타겟 존재 여부 ---")
    if n_pos == 0:
        print("  ⚠️ pos 0개 -> 모델이 []만 학습. always-[] 붕괴의 직접 원인. 데이터 재빌드 필요.")
    elif n_bad > total * 0.05:
        print(f"  ⚠️ bad {n_bad}개({n_bad/total:.1%}) -> target 형식 손상 의심. 샘플 확인 필요.")
    else:
        print(f"  ✓ pos {n_pos}개 정상 존재(형식 이상 {n_bad}개).")

    if pos_scores:
        print("\n--- #3 hook_score 분포(pos) ---")
        distinct = len(set(round(s, 4) for s in pos_scores))
        sd = statistics.pstdev(pos_scores) if len(pos_scores) > 1 else 0.0
        print(f"  min={min(pos_scores):.4f} max={max(pos_scores):.4f} "
              f"mean={statistics.mean(pos_scores):.4f} median={statistics.median(pos_scores):.4f} std={sd:.4f}")
        print(f"  distinct 값 수: {distinct} / {len(pos_scores)} (적으면 고정라벨=붕괴유발)")
        if n_highlights:
            print(f"  highlights 개수: min={min(n_highlights)} max={max(n_highlights)} "
                  f"mean={statistics.mean(n_highlights):.2f}")
        histogram(pos_scores)
        if distinct <= 3:
            print("  ⚠️ hook_score가 사실상 고정값 -> 메모리 B-2(고정 0.9/0.1 붕괴) 해당. 상대값 재라벨 검토.")

    print(f"\n--- #4 target 원문 샘플(각 {args.samples}) ---")
    print(" [pos]")
    for s in pos_samples:
        print(f"   {s}")
    print(" [neg]")
    for s in neg_samples:
        print(f"   {s}")


if __name__ == "__main__":
    main()