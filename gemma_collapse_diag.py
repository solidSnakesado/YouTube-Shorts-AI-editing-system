# 44일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_collapse_diag.py
#
# 목적: round3 checkpoint-100 붕괴(pos/neg 전부 0.82~0.92) 원인 2갈래 확인.
#   (1) base 모델(어댑터 X) 출력 -> base도 0.9면 어댑터 탓 아님(base 경향).
#       base가 다양/낮으면 학습이 0.9로 뭉갠 것(학습 붕괴) -> 처방 갈림.
#   (2) train pos hook_score 실분포 -> pos가 0.8~1.0에 과밀집인지 정밀 측정
#       (collapse_check는 5개 표본만 -> 여기선 전체 pos 집계).
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py(infer/select/parse 재활용).
# 실행(학습 중단 후, VRAM 확보 상태):
#   python gemma_collapse_diag.py --jsonl datasets/gemma_audio/eval_relabel.jsonl \
#       --train datasets/gemma_audio/train_relabel.jsonl
from __future__ import annotations

import argparse
import json
import statistics

from gemma_collapse_check import _stats, infer, parse_hook_score, select_samples


def load_base(base_model: str):
    """어댑터 없이 base(bf16)만 로드 -> (model, processor). PeftModel 미부착."""
    import torch
    import transformers
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def pos_score_dist(train_path: str, thr: float) -> None:
    """train의 pos(=label 외 or gt>=thr) hook_score 전체 분포 집계 + 히스토그램."""
    scores: list[float] = []
    n_pos = n_neg = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("metadata", {}).get("label")
            try:
                from gemma_collate import parse_sample
                gt = parse_hook_score(parse_sample(row)["target"])
            except Exception:                       # noqa: BLE001
                continue
            is_pos = (label != "negative") if label else (gt >= thr)
            if is_pos:
                n_pos += 1
                if gt >= 0:
                    scores.append(gt)
            else:
                n_neg += 1

    print(f"\n=== train pos hook_score 분포 (pos {n_pos} / neg {n_neg}) ===")
    if not scores:
        print("pos 점수 없음")
        return
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"pos 점수: min={min(scores):.4f} max={max(scores):.4f} "
          f"mean={statistics.mean(scores):.4f} std={sd:.4f}")
    buckets = [0] * 10
    for s in scores:
        idx = min(int(s * 10), 9)
        buckets[idx] += 1
    mx = max(buckets) or 1
    for i, c in enumerate(buckets):
        bar = "#" * int(40 * c / mx)
        print(f"  [{i/10:.1f}~{(i+1)/10:.1f})  {c:4d} {bar}")
    # 과밀집 진단: 상위 2버킷(0.8~1.0) 비율
    hi = buckets[8] + buckets[9]
    print(f"  -> 0.8~1.0 구간: {hi}/{len(scores)} = {hi/len(scores):.1%} "
          f"({'과밀집(붕괴유발 가능)' if hi/len(scores) > 0.7 else '분산됨'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="붕괴 원인 진단(base 출력 + pos 분포)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_relabel.jsonl",
                    help="base 추론용 eval(pos/neg 표본 추출)")
    ap.add_argument("--train", default="datasets/gemma_audio/train_relabel.jsonl",
                    help="pos 분포 집계용 train")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--n", type=int, default=5, help="base 추론 pos/neg 표본 수")
    ap.add_argument("--pos-threshold", type=float, default=0.5)
    args = ap.parse_args()

    # (2) 먼저 데이터 분포(모델 로드 전 - 빠름)
    pos_score_dist(args.train, args.pos_threshold)

    # (1) base 모델 출력
    pos_rows, neg_rows = select_samples(args.jsonl, args.n, args.pos_threshold)
    print(f"\n=== base 모델(어댑터 X) 로드: {args.base_model} ===")
    model, processor = load_base(args.base_model)

    print("=== base pos 추론(어댑터 없이) ===")
    pos_pred: list[float] = []
    for i, row in enumerate(pos_rows):
        p, raw = infer(model, processor, row, args.base_dir)
        pos_pred.append(p)
        print(f"  pos[{i}] pred={p:+.4f}  raw={raw}")

    print("=== base neg 추론(어댑터 없이) ===")
    neg_pred: list[float] = []
    for i, row in enumerate(neg_rows):
        p, raw = infer(model, processor, row, args.base_dir)
        neg_pred.append(p)
        print(f"  neg[{i}] pred={p:+.4f}  raw={raw}")

    pn, pmean, _pmin, _pmax = _stats(pos_pred)
    nn, nmean, _nmin, _nmax = _stats(neg_pred)
    print("-" * 56)
    print(f"base pos: n={pn} mean={pmean}")
    print(f"base neg: n={nn} mean={nmean}")
    if pmean is not None and nmean is not None:
        print(f"base 분리도(pos-neg)={pmean - nmean:+.4f}")
        print("해석: base가 이미 0.9 뭉침=base 경향(어댑터 무력) / "
              "base가 다양·낮음=학습이 0.9로 붕괴시킴")
    print("-" * 56)


if __name__ == "__main__":
    main()