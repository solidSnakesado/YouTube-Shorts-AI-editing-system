#!/usr/bin/env python3
"""
51일차: 피드백 -> 기존 v2 train 병합 (eval 고정, round12와 Spearman 직접 비교 유지)

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_fb_merge.py
- 수정 이력: 1차 작성 (51일차) -> 수정 1회(52일차): --feedback 다중 파일(nargs='+') +
  기본 [r1, r2] 누적 + 기본 출력 train_round3 (수정본 기준 L20, L35~38, L78~79, L85~96, L122~123)

[설계 근거]
- round12는 datasets/gemma_audio_v2/{train,eval}_qwenfmt.jsonl로 학습/평가됨 (49일차)
- 2차 학습에서 eval을 재분리하면 round12의 Spearman 0.2671과 비교 불가
  -> eval_qwenfmt.jsonl은 절대 불변, train_qwenfmt.jsonl + 피드백만 병합
- 누수 검사: 피드백 metadata.yt_id가 eval 영상(오디오 경로 앞 11자 YouTube ID)과
  겹치면 해당 샘플 제외
- 업샘플 기본 2배 (35일차 교훈: 과업샘플=평균 회귀 -> 2~3 권장)

[출력]
  기본 datasets/gemma_audio_v2/train_round3.jsonl (기존 파일 무수정 — 비파괴)

[실행]
  python3 scripts/gemma_fb_merge.py
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_TRAIN = "datasets/gemma_audio_v2/train_qwenfmt.jsonl"
DEFAULT_EVAL = "datasets/gemma_audio_v2/eval_qwenfmt.jsonl"
# 52일차: 누적 피드백 기본 [r1, r2] — 3차 학습은 전 사이클 피드백을 모두 포함
DEFAULT_FB = ["datasets/gemma_audio/dataset_feedback_r1.jsonl",
              "datasets/gemma_audio/dataset_feedback_r2.jsonl"]
DEFAULT_OUT = "datasets/gemma_audio_v2/train_round3.jsonl"    # 52일차: 기본 r3
DEFAULT_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """51일차: jsonl 로드 (파싱 실패 시 즉시 중단 — 학습 입력이므로 무결성 우선)"""

    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"파싱 실패: {path}:{i}")
    return rows


def video_id_of(sample: dict) -> str:
    """51일차: 오디오 경로 stem 앞 11자 = YouTube ID (package_dataset 43일차 규칙 동일)"""

    content = sample["messages"][0]["content"]
    audio = next((b["audio"] for b in content if b.get("type") == "audio"), "")
    return Path(audio).stem[:11]


def target_score(sample: dict) -> float | None:
    """51일차: 타겟 텍스트 첫 숫자 (collate extract_score와 동일 방식, 검증용)"""

    import re
    m = re.search(r"[-+]?\d*\.?\d+", sample["messages"][1]["content"][0]["text"])
    return float(m.group()) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="피드백 -> v2 train 병합 (eval 고정)")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--eval", dest="eval_path", default=DEFAULT_EVAL)
    # 52일차: 다중 피드백 파일 (누적 사이클)
    ap.add_argument("--feedback", nargs="+", default=DEFAULT_FB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--upsample", type=int, default=2, help="피드백 업샘플 배수 (기본 2)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    # 52일차: 피드백 다중화 — train/eval + 피드백 목록 각각 실존 검증
    paths = {"train": Path(args.train), "eval": Path(args.eval_path)}
    fb_paths = [Path(p) for p in args.feedback]
    for name, p in list(paths.items()) + [(f"feedback[{i}]", p) for i, p in enumerate(fb_paths)]:
        if not p.is_file():
            print(f"파일 없음 ({name}): {p}")
            return 1

    train_rows = load_jsonl(paths["train"])
    eval_rows = load_jsonl(paths["eval"])
    fb_per_file = [(p, load_jsonl(p)) for p in fb_paths]        # 52일차: 파일별 보존(리포트)
    fb_rows = [s for _, rows in fb_per_file for s in rows]

    # 51일차: 누수 검사 - eval 영상 YouTube ID vs 피드백 yt_id
    eval_vids = {video_id_of(s) for s in eval_rows}
    kept, leaked = [], []
    for s in fb_rows:
        yt = str(s.get("metadata", {}).get("yt_id", ""))
        if yt and yt in eval_vids:
            leaked.append(yt)
        else:
            kept.append(s)

    up = max(1, args.upsample)
    merged = train_rows + kept * up
    random.Random(args.seed).shuffle(merged)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 51일차: 리포트 - 타겟 분포(피드백)와 클래스 확인
    fb_scores = [t for t in (target_score(s) for s in kept) if t is not None]
    fb_labels = Counter(s.get("metadata", {}).get("feedback", "?") for s in kept)
    print("=" * 60)
    for p, rows in fb_per_file:                                 # 52일차: 파일별 행수
        print(f"피드백 입력: {p} ({len(rows)}행)")
    print(f"train(기존) {len(train_rows)} + 피드백 {len(kept)}x{up} = {len(merged)}행 -> {out}")
    print(f"eval 고정: {paths['eval']} ({len(eval_rows)}행, 무수정)")
    print(f"누수 제외: {len(leaked)}건" + (f" ({sorted(set(leaked))})" if leaked else ""))
    print(f"피드백 라벨: OK {fb_labels.get('OK', 0)} / NO {fb_labels.get('NO', 0)}")
    if fb_scores:
        print(f"피드백 타겟: min {min(fb_scores):.3f} / mean {sum(fb_scores)/len(fb_scores):.3f} "
              f"/ max {max(fb_scores):.3f}")
    ok = len(merged) == len(train_rows) + len(kept) * up
    print("종합: " + ("병합 완료 ✓" if ok else "행수 불일치 ⚠️"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())