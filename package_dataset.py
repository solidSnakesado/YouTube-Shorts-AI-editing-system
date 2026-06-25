#!/usr/bin/env python3
# 계층: 패키징 유틸 (루트 실행 스크립트)
# 역할: 포지티브+네거티브 병합 -> 시드 셔플 -> 학습용 단일 jsonl + 검증
# 39일차 신규: dataset.jsonl + dataset_neg.jsonl -> all.jsonl (Colab 업로드용)
#   - 셔플 필수: 블록 배치(전부 pos 후 전부 neg)면 배치 불균형 -> 시드 셔플로 교차
#   - 검증: 총행수 = pos+neg, pos/neg 분포, 최대 동일라벨 연속(셔플 품질), 파싱오류
#   - 비파괴: 원본 2개 보존, --out에만 기록. 시드 고정(--seed)으로 재현 가능

"""포지티브/네거티브 병합·셔플 패키징 CLI."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_POS = "datasets/gemma_audio/dataset.jsonl"
DEFAULT_NEG = "datasets/gemma_audio/dataset_neg.jsonl"
DEFAULT_OUT = "datasets/gemma_audio/all.jsonl"
DEFAULT_SEED = 42


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
    """assistant 출력 -> 'pos' | 'neg' | 'bad'."""

    try:
        highlights = json.loads(sample["messages"][1]["content"][0]["text"]).get("highlights")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "bad"
    if highlights == []:
        return "neg"
    if isinstance(highlights, list) and len(highlights) == 1 and "hook_score" in highlights[0]:
        return "pos"
    return "bad"


def max_same_run(labels) -> int:
    """동일 라벨 최대 연속 길이 (셔플 품질 지표)."""

    best = run = 0
    prev = None
    for x in labels:
        run = run + 1 if x == prev else 1
        best = max(best, run)
        prev = x
    return best


def _parse_args():
    ap = argparse.ArgumentParser(description="포지티브/네거티브 병합·셔플 패키징")
    ap.add_argument("--pos-path", default=DEFAULT_POS, help="포지티브 jsonl")
    ap.add_argument("--neg-path", default=DEFAULT_NEG, help="네거티브 jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="병합·셔플 출력 경로")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="셔플 시드(재현용)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    pos_path, neg_path, out_path = Path(args.pos_path), Path(args.neg_path), Path(args.out)
    for p in (pos_path, neg_path):
        if not p.exists():
            print(f"파일 없음: {p}")
            return 1

    pos_rows, pos_err = load_jsonl(pos_path)
    neg_rows, neg_err = load_jsonl(neg_path)
    merged = pos_rows + neg_rows
    random.seed(args.seed)
    random.shuffle(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in merged:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 검증
    kinds = Counter(classify(s) for s in merged)
    labels = [classify(s) for s in merged]
    run = max_same_run(labels)
    head = "".join("P" if x == "pos" else ("N" if x == "neg" else "?") for x in labels[:30])

    print("=== 데이터 패키징 (병합·셔플) ===")
    print(f"포지티브: {len(pos_rows)} (파싱오류 {len(pos_err)}) | "
          f"네거티브: {len(neg_rows)} (파싱오류 {len(neg_err)})")
    print(f"출력: {out_path} | 총 {len(merged)}행 (시드 {args.seed})")
    print(f"\n[분포] 포지티브 {kinds['pos']} | 네거티브 {kinds['neg']} | 불량 {kinds['bad']}")
    print(f"[셔플 품질] 최대 동일라벨 연속: {run} "
          f"({'양호' if run < 20 else '주의: 블록 의심'})")
    print(f"[앞 30개 라벨] {head}")

    integrity = (len(merged) == len(pos_rows) + len(neg_rows)
                 and kinds["bad"] == 0 and not pos_err and not neg_err)
    print("\n=== 종합: " + ("패키징 완료 (Colab 업로드 준비)" if integrity
                          else "문제 발견 (위 항목 확인)") + " ===")
    return 0 if integrity else 1


if __name__ == "__main__":
    sys.exit(main())