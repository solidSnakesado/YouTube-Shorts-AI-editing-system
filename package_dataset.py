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


def video_id_of(sample: dict) -> str:
    """43일차: 샘플의 오디오 경로에서 영상 ID(YouTube 11자) 추출.

    같은 영상의 클립을 train/eval 한쪽으로만 묶기 위함(클립 무작위 분리=누수).
    파일명 형식 {video_id}_{clip} 이고 YouTube ID는 11자, 12번째가 구분자 '_'.
    """
    content = sample["messages"][0]["content"]
    audio = next((b["audio"] for b in content if b.get("type") == "audio"), "")
    return Path(audio).stem[:11]


def video_level_split(rows, eval_ratio, seed):
    """43일차: 영상 단위 stratified train/eval 분리.

    영상을 셔플 후 eval 의 pos/neg 표본수가 각각 목표(eval_ratio)에 도달할 때까지
    통째로 eval 에 배정 -> 영상 누수 0 + pos/neg 균형 근사. 반환 (train, eval, info).
    """
    by_vid: dict = {}
    for r in rows:
        by_vid.setdefault(video_id_of(r), []).append(r)
    stats = {v: Counter(classify(s) for s in rs) for v, rs in by_vid.items()}
    tot_pos = sum(c["pos"] for c in stats.values())
    tot_neg = sum(c["neg"] for c in stats.values())

    vids = list(by_vid.keys())
    rng = random.Random(seed)
    rng.shuffle(vids)
    ep_t, en_t = tot_pos * eval_ratio, tot_neg * eval_ratio
    ep = en = 0
    eval_vids = set()
    for v in vids:
        if ep >= ep_t and en >= en_t:
            break
        eval_vids.add(v)
        ep += stats[v]["pos"]
        en += stats[v]["neg"]

    eval_rows = [s for v in eval_vids for s in by_vid[v]]
    train_rows = [s for v in vids if v not in eval_vids for s in by_vid[v]]
    rng.shuffle(train_rows)          # split 내부도 셔플(pos/neg 교차 -> 배치 균형)
    rng.shuffle(eval_rows)
    info = {
        "videos": len(vids), "eval_videos": len(eval_vids),
        "train": Counter(classify(s) for s in train_rows),
        "eval": Counter(classify(s) for s in eval_rows),
        "overlap": eval_vids & {video_id_of(s) for s in train_rows},
    }
    return train_rows, eval_rows, info


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _parse_args():
    ap = argparse.ArgumentParser(description="포지티브/네거티브 병합·셔플 패키징")
    ap.add_argument("--pos-path", default=DEFAULT_POS, help="포지티브 jsonl")
    ap.add_argument("--neg-path", default=DEFAULT_NEG, help="네거티브 jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="병합·셔플 출력 경로")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="셔플 시드(재현용)")
    ap.add_argument("--eval-ratio", type=float, default=0.2,
                    help="영상 단위 eval 분리 비율(0이면 분리 안 함). 기본 0.2")
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
    _write_jsonl(out_path, merged)

    # 43일차: 영상 단위 stratified split (클립 누수 방지). eval_ratio>0 일 때만.
    split_ok = True
    if args.eval_ratio > 0:
        train_rows, eval_rows, sinfo = video_level_split(merged, args.eval_ratio, args.seed)
        train_path = out_path.parent / "train.jsonl"
        eval_path = out_path.parent / "eval.jsonl"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(eval_path, eval_rows)
        n = len(merged)
        e_ratio = len(eval_rows) / n if n else 0
        print("=== 영상 단위 split (누수 방지) ===")
        print(f"총 영상 {sinfo['videos']} -> eval 영상 {sinfo['eval_videos']}")
        print(f"train: {train_path} | {len(train_rows)}행 "
              f"(pos {sinfo['train']['pos']} / neg {sinfo['train']['neg']})")
        print(f"eval : {eval_path} | {len(eval_rows)}행 "
              f"(pos {sinfo['eval']['pos']} / neg {sinfo['eval']['neg']}) "
              f"= 실제 {e_ratio*100:.1f}%")
        if sinfo["overlap"]:
            split_ok = False
            print(f"⚠️ 영상 누수 발견: {len(sinfo['overlap'])}개 train/eval 양쪽 -> 분리 실패")
        else:
            print("영상 누수: 0 (train/eval 영상 겹침 없음) ✓")
        print()

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
                 and kinds["bad"] == 0 and not pos_err and not neg_err and split_ok)
    print("\n=== 종합: " + ("패키징 완료 (Colab 업로드 준비)" if integrity
                          else "문제 발견 (위 항목 확인)") + " ===")
    return 0 if integrity else 1


if __name__ == "__main__":
    sys.exit(main())