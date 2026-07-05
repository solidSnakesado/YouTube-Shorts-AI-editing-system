# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_class_check.py
#
# 목적: 분류 버킷(2클래스) 재학습 checkpoint의 붕괴 검사. 회귀용
#   gemma_collapse_check.py(점수 분리도)와 달리, 분류는 "상"/"하" 등급 토큰의
#   혼동행렬/정확도/쏠림으로 판정. 핵심 붕괴 신호:
#     - 쏠림: 전부 "상" 또는 전부 "하"(한 클래스만 출력) -> 붕괴
#     - 형식 미학습: "상"/"하" 둘 다 아닌 출력 다수(None) -> 붕괴
#     - 건강: pos는 대부분 "상", neg는 대부분 "하"(각 정확도 60%+)
#   loss는 붕괴 못 잡음 -> 본 검사가 1차 지표(회귀와 동일 철학).
#
# 재사용: load_checkpoint/infer는 gemma_collapse_check에서 그대로(중복 제거).
#   infer가 반환하는 raw 텍스트만 parse_class로 재파싱(점수 파싱 결과는 버림).
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py. (본 파일도 추가 업로드)
# 실행(학습과 다른 셀, checkpoint 저장 후):
#   python gemma_class_check.py --checkpoint /content/drive/MyDrive/gemma4_adapters/<round>_ckpt/checkpoint-100
#   python gemma_class_check.py --checkpoint <경로> --n 8
from __future__ import annotations

import argparse
import json
from typing import Optional

from gemma_collapse_check import infer, load_checkpoint

BASE_MODEL = "unsloth/gemma-4-E4B-it"


def parse_class(text: str, pos_lbl: str, neg_lbl: str) -> Optional[str]:
    """생성 텍스트 -> 'pos'/'neg'/None. 맨 앞 라벨 우선, 폴백 단독 등장.
    둘 다 또는 둘 다 아니면(모호/형식이탈) None."""
    s = text.strip()
    if s.startswith(pos_lbl):
        return "pos"
    if s.startswith(neg_lbl):
        return "neg"
    has_pos, has_neg = pos_lbl in s, neg_lbl in s
    if has_pos and not has_neg:
        return "pos"
    if has_neg and not has_pos:
        return "neg"
    return None


def select_by_label(jsonl: str, n: int):
    """metadata.label 기준 pos n / neg n 선택(데이터 구조: negative만 label 보유).
    target이 상/하라 점수 파싱은 안 함."""
    pos: list[dict] = []
    neg: list[dict] = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            is_pos = row.get("metadata", {}).get("label") != "negative"
            if is_pos and len(pos) < n:
                pos.append(row)
            elif (not is_pos) and len(neg) < n:
                neg.append(row)
            if len(pos) >= n and len(neg) >= n:
                break
    return pos, neg


def _run(model, processor, rows, base_dir, pos_lbl, neg_lbl, tag):
    """rows 추론 -> 분류 결과 리스트('pos'/'neg'/None) + raw 출력."""
    results: list[Optional[str]] = []
    for i, row in enumerate(rows):
        _, raw = infer(model, processor, row, base_dir)   # 점수는 버리고 raw만
        cls = parse_class(raw, pos_lbl, neg_lbl)
        results.append(cls)
        print(f"  {tag}[{i}] pred={cls!s:>4}  raw={raw!r}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint 붕괴 검사(2클래스 혼동행렬)")
    ap.add_argument("--checkpoint", required=True, help="검사할 checkpoint-* 경로")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_binclass.jsonl")
    ap.add_argument("--n", type=int, default=8, help="pos/neg 각 표본 수")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-dir", default=None, help="미디어 상대경로 기준(보통 None=cwd)")
    ap.add_argument("--pos-label", default="상")
    ap.add_argument("--neg-label", default="하")
    args = ap.parse_args()

    pos_rows, neg_rows = select_by_label(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)} (요청 {args.n})")

    print(f"=== 로드: {args.checkpoint} (베이스 {args.base_model}, bf16) ===")
    model, processor = load_checkpoint(args.checkpoint, args.base_model)

    print(f"=== pos 추론(정답 '{args.pos_label}') ===")
    pos_res = _run(model, processor, pos_rows, args.base_dir,
                   args.pos_label, args.neg_label, "pos")
    print(f"=== neg 추론(정답 '{args.neg_label}') ===")
    neg_res = _run(model, processor, neg_rows, args.base_dir,
                   args.pos_label, args.neg_label, "neg")

    pn, nn = len(pos_res), len(neg_res)
    pos_ok = pos_res.count("pos")          # pos를 pos로(정답)
    neg_ok = neg_res.count("neg")          # neg를 neg로(정답)
    none_cnt = pos_res.count(None) + neg_res.count(None)
    pred_pos = pos_res.count("pos") + neg_res.count("pos")
    pred_neg = pos_res.count("neg") + neg_res.count("neg")
    total = pn + nn

    print("-" * 56)
    print(f"혼동행렬: pos->[{args.pos_label}{pos_res.count('pos')}/"
          f"{args.neg_label}{pos_res.count('neg')}/None{pos_res.count(None)}]  "
          f"neg->[{args.pos_label}{neg_res.count('pos')}/"
          f"{args.neg_label}{neg_res.count('neg')}/None{neg_res.count(None)}]")
    pos_acc = pos_ok / pn if pn else 0
    neg_acc = neg_ok / nn if nn else 0
    acc = (pos_ok + neg_ok) / total if total else 0
    print(f"정확도: 전체 {acc:.1%} (pos {pos_acc:.1%}, neg {neg_acc:.1%}) "
          f"| 예측분포 {args.pos_label}{pred_pos}/{args.neg_label}{pred_neg}/None{none_cnt}")
    print("-" * 56)

    if total and none_cnt >= total * 0.5:
        print(f"판정: 붕괴(형식 미학습) — None {none_cnt}/{total}. '{args.pos_label}'/"
              f"'{args.neg_label}' 토큰을 못 냄. 중단+knob/지시문 검토")
    elif pred_pos == 0 or pred_neg == 0:
        only = args.pos_label if pred_pos else args.neg_label
        print(f"판정: 붕괴(쏠림) — 전부 '{only}' 한쪽으로. 입력무관 상수. 중단+knob 조정")
    elif pos_acc >= 0.6 and neg_acc >= 0.6:
        print(f"판정: 건강한 분리 — pos '{args.pos_label}'/neg '{args.neg_label}' "
              f"각 60%+ (전체 {acc:.1%}). 학습 계속 진행 가능")
    else:
        print(f"판정: 애매(부분 분리, 전체 {acc:.1%}) — 위 혼동행렬로 직접 판단")


if __name__ == "__main__":
    main()