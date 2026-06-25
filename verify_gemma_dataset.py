#!/usr/bin/env python3
# 계층: 검증 유틸 (루트 실행 스크립트)
# 역할: Gemma 데이터셋(dataset.jsonl / dataset_neg.jsonl / 병합본) 단독 품질 검증
# 39일차 신규: 포지티브 풀빌드 직후 품질 확인용. pos/neg 자동 판별 -> 네거티브·병합본 재사용
#   검증: (A) JSON 무결성 (B) messages 스키마 (C) content-block 키(Unsloth 대조)
#         (D) hook_score graded 분포 (E) 물리 파일(프레임/오디오) 실존 (F) 메타 정합성
#   미디어 경로는 cwd 기준 상대경로 -> 프로젝트 루트(~/project/yt_shorts_ai)에서 실행할 것

"""Gemma 데이터셋 품질 단독 검증 CLI."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# 39일차: content-block 키 - Unsloth Gemma 오디오 노트북 로더 규격과 대조 대상
EXPECTED_FRAME_KEY = "image"
EXPECTED_AUDIO_KEY = "audio"
DEFAULT_PATH = "datasets/gemma_audio/dataset.jsonl"


def load_jsonl(path: Path):
    """jsonl 로드 - 파싱 실패 행은 (행번호, 오류) 리스트로 분리 수집."""

    rows, errors = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append((i, str(exc)))
    return rows, errors


def classify(sample: dict):
    """assistant 출력 파싱 -> ('pos', hook_score) | ('neg', None) | ('bad', None)."""

    try:
        text = sample["messages"][1]["content"][0]["text"]
        obj = json.loads(text)
        highlights = obj.get("highlights", None)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "bad", None
    if highlights == []:
        return "neg", None
    if isinstance(highlights, list) and len(highlights) == 1 and "hook_score" in highlights[0]:
        return "pos", highlights[0]["hook_score"]
    return "bad", None


def check_schema(sample: dict):
    """messages 구조 + content-block 키 검증 -> 오류 메시지 리스트."""

    errs = []
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        return ["messages가 2개 턴이 아님"]
    user, asst = msgs[0], msgs[1]
    if user.get("role") != "user":
        errs.append("0번 턴 role != user")
    if asst.get("role") != "assistant":
        errs.append("1번 턴 role != assistant")
    content = user.get("content", [])
    if not isinstance(content, list) or not content:
        return errs + ["user content 비어있음"]
    n_img = sum(1 for b in content if b.get("type") == "image")
    n_aud = sum(1 for b in content if b.get("type") == "audio")
    n_txt = sum(1 for b in content if b.get("type") == "text")
    if n_img < 1:
        errs.append("image 블록 없음")
    if n_aud != 1:
        errs.append(f"audio 블록 {n_aud}개 (1개 기대)")
    if n_txt != 1:
        errs.append(f"text 블록 {n_txt}개 (1개 기대)")
    # content-block 키 존재 검증 (Unsloth 로더 대조)
    if any(b.get("type") == "image" and EXPECTED_FRAME_KEY not in b for b in content):
        errs.append(f"image 블록에 '{EXPECTED_FRAME_KEY}' 키 없음")
    if any(b.get("type") == "audio" and EXPECTED_AUDIO_KEY not in b for b in content):
        errs.append(f"audio 블록에 '{EXPECTED_AUDIO_KEY}' 키 없음")
    return errs


def media_paths(sample: dict):
    """user content -> (frame 경로 리스트, audio 경로). 구조 불량 시 ([], None)."""

    try:
        content = sample["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return [], None
    frames, audio = [], None
    for b in content:
        if b.get("type") == "image":
            frames.append(b.get(EXPECTED_FRAME_KEY))
        elif b.get("type") == "audio":
            audio = b.get(EXPECTED_AUDIO_KEY)
    return frames, audio


def check_files(sample: dict, base: Path):
    """프레임/오디오 실존 -> (missing_frame_수, audio_ok)."""

    frames, audio = media_paths(sample)
    miss = sum(1 for p in frames if not p or not (base / p).exists())
    audio_ok = bool(audio) and (base / audio).exists()
    return miss, audio_ok


def _parse_args():
    ap = argparse.ArgumentParser(description="Gemma 데이터셋 품질 검증")
    ap.add_argument("--path", default=DEFAULT_PATH, help="검증할 jsonl 경로")
    ap.add_argument("--base-dir", default=".", help="미디어 상대경로 기준 (기본 cwd)")
    ap.add_argument("--max-file-checks", type=int, default=0,
                    help="미디어 실존 검사 샘플 수 (0=전수, 기본)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    path, base = Path(args.path), Path(args.base_dir)
    if not path.exists():
        print(f"파일 없음: {path}")
        return 1

    rows, parse_errs = load_jsonl(path)
    print(f"=== Gemma 데이터셋 품질 검증: {path} ===")
    print(f"총 행: {len(rows)} | JSON 파싱 실패: {len(parse_errs)}")
    for ln, exc in parse_errs[:5]:
        print(f"  - L{ln}: {exc}")

    # 분류 + 스키마
    kinds = Counter()
    hooks = []
    schema_fail = []
    for idx, sample in enumerate(rows):
        kind, hook = classify(sample)
        kinds[kind] += 1
        if kind == "pos":
            hooks.append(hook)
        errs = check_schema(sample)
        if errs:
            schema_fail.append((idx, errs))

    print(f"\n[분류] 포지티브 {kinds['pos']} | 네거티브 {kinds['neg']} | 불량 {kinds['bad']}")
    print(f"[스키마] 통과 {len(rows) - len(schema_fail)} / 실패 {len(schema_fail)}")
    for idx, errs in schema_fail[:5]:
        print(f"  - row {idx}: {', '.join(errs)}")

    # hook_score graded 분포 (단일값 붕괴 점검)
    if hooks:
        distinct = len({round(h, 4) for h in hooks})
        print(f"\n[hook_score] n={len(hooks)} min={min(hooks):.4f} "
              f"max={max(hooks):.4f} mean={sum(hooks) / len(hooks):.4f} distinct={distinct}")
        oor = [h for h in hooks if not 0.0 <= h <= 1.0]
        if oor:
            print(f"  주의: 범위(0~1) 벗어남 {len(oor)}개")
        print("  graded 분포 (붕괴 아님)" if distinct > 1 else "  주의: distinct=1 단일값 붕괴 의심")

    # 메타 정합성
    meta_fail = sum(
        1 for s in rows
        if not all(k in s.get("metadata", {}) for k in ("video_id", "clip_start", "clip_end"))
    )
    labels = Counter(s.get("metadata", {}).get("label", "(없음)") for s in rows)
    print(f"\n[메타] video_id/clip_start/clip_end 누락 행: {meta_fail}")
    print(f"[메타 label] {dict(labels)}")

    # 물리 파일 실존
    if args.max_file_checks and args.max_file_checks < len(rows):
        targets = random.sample(rows, args.max_file_checks)
        scope = f"{len(targets)}개 샘플"
    else:
        targets = rows
        scope = "전수"
    miss_frame_total = samples_with_missing = audio_miss = 0
    for sample in targets:
        miss, audio_ok = check_files(sample, base)
        if miss:
            miss_frame_total += miss
            samples_with_missing += 1
        if not audio_ok:
            audio_miss += 1
    print(f"\n[물리 파일/{scope}] 프레임 누락 {miss_frame_total}개"
          f"(샘플 {samples_with_missing}개) | 오디오 누락 {audio_miss}개")

    # 종합
    ok = (not parse_errs and not schema_fail and kinds["bad"] == 0
          and meta_fail == 0 and miss_frame_total == 0 and audio_miss == 0)
    print("\n=== 종합: " + ("전체 통과" if ok else "문제 발견 (위 항목 확인)") + " ===")
    print("참고: 빌드 로그의 '오디오없음 스킵'/'피크 다운로드 실패'는 "
          "의도적 누락(무손실 추적값)이며 본 검증과 별개입니다.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())