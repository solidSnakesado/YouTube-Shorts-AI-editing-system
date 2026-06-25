#!/usr/bin/env python3
# 계층: 복구 유틸 (루트 실행 스크립트)
# 역할: 네거티브 초과 매칭 복구 — 영상별 네거티브를 pos_count까지 트림(첫 처리분 유지)
# 39일차 신규: 히트맵 중복으로 네거티브가 일부 영상 2배 생성된 데이터 복구
#   - 영상별 pos 수 집계(dataset.jsonl) -> neg를 pos 수까지만 유지(파일 순서=첫 처리)
#   - 초과분 드롭 + 그 고아 미디어(프레임 디렉토리/오디오) 식별
#   - 비파괴: 복구본을 --out(기본 dataset_neg.repaired.jsonl)에 기록, 원본 보존
#   - 미달(neg<pos)은 그대로 둠(오디오 스킵 등 본질적 한계). 미디어 경로는 cwd 기준 상대경로

"""네거티브 초과 매칭 복구 CLI (트림 + 고아 미디어 정리)."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_POS = "datasets/gemma_audio/dataset.jsonl"
DEFAULT_NEG = "datasets/gemma_audio/dataset_neg.jsonl"
DEFAULT_OUT = "datasets/gemma_audio/dataset_neg.repaired.jsonl"
MEDIA_ROOT = "datasets/gemma_audio"   # 삭제 안전장치: 이 하위 경로만 허용


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def count_positives(path: Path) -> Counter:
    """dataset.jsonl에서 영상별 포지티브 수 집계."""

    counts = Counter()
    for sample in load_jsonl(path):
        if classify(sample) == "pos":
            vid = sample.get("metadata", {}).get("video_id")
            if vid:
                counts[vid] += 1
    return counts


def media_of(sample: dict):
    """샘플 -> (프레임 디렉토리 집합, 오디오 경로|None)."""

    frame_dirs, audio = set(), None
    try:
        content = sample["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return frame_dirs, None
    for block in content:
        if block.get("type") == "image" and block.get("image"):
            frame_dirs.add(str(Path(block["image"]).parent))
        elif block.get("type") == "audio":
            audio = block.get("audio")
    return frame_dirs, audio


def collect_media(samples):
    """샘플 리스트 -> (프레임 디렉토리 집합, 오디오 경로 집합)."""

    dirs, auds = set(), set()
    for s in samples:
        d, a = media_of(s)
        dirs |= d
        if a:
            auds.add(a)
    return dirs, auds


def _under_root(path: Path, root: Path) -> bool:
    """path가 root 하위인지(삭제 안전장치)."""

    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _parse_args():
    ap = argparse.ArgumentParser(description="네거티브 초과 매칭 복구")
    ap.add_argument("--pos-path", default=DEFAULT_POS, help="포지티브 jsonl")
    ap.add_argument("--neg-path", default=DEFAULT_NEG, help="네거티브 jsonl(원본)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="복구본 출력 경로(비파괴)")
    ap.add_argument("--base-dir", default=".", help="미디어 상대경로 기준(기본 cwd)")
    ap.add_argument("--delete-orphans", action="store_true",
                    help="고아 미디어 실제 삭제(기본: 목록만 출력)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    pos_path, neg_path, out_path = Path(args.pos_path), Path(args.neg_path), Path(args.out)
    base = Path(args.base_dir)
    for p in (pos_path, neg_path):
        if not p.exists():
            print(f"파일 없음: {p}")
            return 1

    pos = count_positives(pos_path)
    neg_rows = load_jsonl(neg_path)

    # 영상별 pos 수까지만 유지(파일 순서=첫 처리분), 초과분 드롭
    kept_count = Counter()
    kept, dropped = [], []
    for sample in neg_rows:
        vid = sample.get("metadata", {}).get("video_id")
        limit = pos.get(vid, 0)
        if vid and kept_count[vid] < limit:
            kept.append(sample)
            kept_count[vid] += 1
        else:
            dropped.append(sample)

    # 트림된 영상 집계
    neg_before = Counter()
    for s in neg_rows:
        vid = s.get("metadata", {}).get("video_id")
        if vid:
            neg_before[vid] += 1
    trimmed = sorted(
        ((vid, neg_before[vid], kept_count[vid]) for vid in neg_before if kept_count[vid] < neg_before[vid]),
        key=lambda x: x[1] - x[2], reverse=True,
    )

    print("=== 네거티브 초과 매칭 복구 ===")
    print(f"포지티브 총: {sum(pos.values())} | 네거티브 원본: {len(neg_rows)}")
    print(f"유지: {len(kept)} | 드롭(초과분): {len(dropped)}")
    print(f"복구 후 네거티브: {len(kept)} (목표≈포지티브 {sum(pos.values())} - 미달분)")
    if trimmed:
        print(f"\n[트림된 영상] {len(trimmed)}개:")
        for vid, before, after in trimmed:
            print(f"  {vid}: {before} -> {after} (드롭 {before - after})")

    # 고아 미디어 = 드롭 샘플 미디어 - 유지 샘플 미디어
    kept_dirs, kept_auds = collect_media(kept)
    drop_dirs, drop_auds = collect_media(dropped)
    orphan_dirs = sorted(drop_dirs - kept_dirs)
    orphan_auds = sorted(drop_auds - kept_auds)
    print(f"\n[고아 미디어] 프레임 디렉토리 {len(orphan_dirs)}개 | 오디오 {len(orphan_auds)}개")

    if args.delete_orphans:
        root = base / MEDIA_ROOT
        d_cnt = a_cnt = 0
        for d in orphan_dirs:
            full = base / d
            if _under_root(full, root) and full.is_dir():
                shutil.rmtree(full)
                d_cnt += 1
        for a in orphan_auds:
            full = base / a
            if _under_root(full, root) and full.is_file():
                full.unlink()
                a_cnt += 1
        print(f"  삭제 완료: 디렉토리 {d_cnt}개, 오디오 {a_cnt}개")
    elif orphan_dirs or orphan_auds:
        print("  (삭제하려면 --delete-orphans 추가. 미삭제 시 Colab 업로드 용량만 증가)")

    # 복구본 기록(비파괴)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\n복구본 기록: {out_path} ({len(kept)}행)")
    print("다음: compare로 검증 후 원본 교체 (mv 복구본 원본)")
    return 0


if __name__ == "__main__":
    sys.exit(main())