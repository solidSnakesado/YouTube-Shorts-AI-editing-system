# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_audio_probe.py
#
# 목적: 분류 2클래스 학습이 round1(전부 상)/round2(거의 전부 하, 45%) 모두 쏠림 붕괴.
#   (a) 출력 형식 카드 소진(회귀5 + 분류 lr/rank 변경 무효) -> (b) 입력 신호 부족 의심.
#   ablation(오디오 유/무 추론) 전, "오디오에 애초에 판별 신호가 있는가"를 데이터로 먼저 확인.
#   무음/거의 무음 클립이 다수면 ablation 이전에 데이터가 원인(오디오 인코더 탓 아님).
#
# 측정(학습/추론 없음, librosa RMS만):
#   1) pos/neg 각 오디오의 RMS 에너지 -> 무음(임계 미만) 비율
#   2) pos vs neg RMS 분포 차이(에너지가 pos/neg를 가르는 신호인지)
#   3) 누락/로드실패 오디오 수
#
# 판정(참고): 무음 비율이 높거나(>30%) pos/neg RMS 분포가 거의 겹치면 -> 오디오에
#   판별 신호 약함(쏠림의 데이터적 원인). 무음 적고 pos/neg RMS 갈리면 -> 신호는 있음,
#   모델이 못 쓰는 것(ablation/인코더 학습으로). 단 RMS는 거친 지표(에너지만, 내용 무관).
#
# 의존: librosa, numpy (로컬 설치됨). gemma_collate 미의존(경로만 직접 파싱).
# 실행(로컬 WSL):
#   python gemma_audio_probe.py
#   python gemma_audio_probe.py --sample 400   # 표본 줄여 빠르게
from __future__ import annotations

import argparse
import json
import os
import statistics as st

SILENCE_RMS = 0.005   # 이 RMS 미만이면 사실상 무음으로 간주(16bit 정규화 기준 경험값)


def audio_path_of(row: dict) -> str | None:
    """messages[0].content의 audio 블록에서 경로 추출(gemma_collate parse와 동일 규약)."""
    try:
        content = row["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    for b in content:
        if b.get("type") == "audio":
            return b.get("audio")
    return None


def rms_of(path: str, base: str) -> float | None:
    """오디오 RMS(전체 평균). 로드 실패/빈 파일 None."""
    import librosa
    import numpy as np

    fp = path if os.path.isabs(path) else os.path.join(base, path)
    if not os.path.exists(fp):
        return None
    try:
        wav, _sr = librosa.load(fp, sr=16000, mono=True)
    except Exception:                               # noqa: BLE001
        return None
    if wav.size == 0:
        return None
    return float(np.sqrt(np.mean(wav.astype("float64") ** 2)))


def collect(jsonl: str, base: str, limit: int):
    """pos/neg별 RMS 리스트 + 누락/실패 카운트 수집(metadata.label 기준)."""
    pos: list[float] = []
    neg: list[float] = []
    miss = fail = 0
    seen_pos = seen_neg = 0
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            is_pos = row.get("metadata", {}).get("label") != "negative"
            if limit > 0:
                if seen_pos >= limit and seen_neg >= limit:
                    break
                if is_pos and seen_pos >= limit:
                    continue
                if (not is_pos) and seen_neg >= limit:
                    continue
            ap = audio_path_of(row)
            if not ap:
                miss += 1
                continue
            r = rms_of(ap, base)
            if r is None:
                fail += 1
                continue
            if is_pos:
                pos.append(r)
                seen_pos += 1
            else:
                neg.append(r)
                seen_neg += 1
    return pos, neg, miss, fail


def describe(name: str, vals: list[float]) -> None:
    """RMS 분포 + 무음 비율 출력."""
    if not vals:
        print(f"{name}: 없음")
        return
    mean = sum(vals) / len(vals)
    med = st.median(vals)
    std = st.pstdev(vals) if len(vals) > 1 else 0.0
    silent = sum(1 for v in vals if v < SILENCE_RMS)
    print(f"{name}: n={len(vals)} mean={mean:.4f} median={med:.4f} std={std:.4f} "
          f"min={min(vals):.4f} max={max(vals):.4f}")
    print(f"        무음(RMS<{SILENCE_RMS}): {silent}/{len(vals)} ({silent / len(vals):.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="binclass 오디오 RMS 실재성 점검")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/train_binclass.jsonl")
    ap.add_argument("--base-dir", default=".", help="오디오 상대경로 기준(보통 저장소 루트)")
    ap.add_argument("--sample", type=int, default=0, help="pos/neg 각 표본 상한(0=전체)")
    args = ap.parse_args()

    print(f"=== {args.jsonl} 오디오 RMS 점검 (표본 {args.sample or '전체'}) ===")
    pos, neg, miss, fail = collect(args.jsonl, args.base_dir, args.sample)
    print(f"누락(audio 경로 없음): {miss} | 로드 실패: {fail}")
    print("-" * 56)
    describe("POS", pos)
    describe("NEG", neg)
    print("-" * 56)

    if pos and neg:
        pm, nm = sum(pos) / len(pos), sum(neg) / len(neg)
        gap = pm - nm
        pooled = (st.pstdev(pos) + st.pstdev(neg)) / 2 or 1e-9
        cohen = gap / pooled                        # 표준화 효과크기(분포 분리 정도)
        print(f"pos-neg RMS 차이: {gap:+.4f} (pos {pm:.4f} / neg {nm:.4f})")
        print(f"효과크기(Cohen's d 근사): {cohen:+.3f} "
              f"(|d|<0.2 무시할 수준, 0.5 중간, 0.8+ 큼)")
        sp = sum(1 for v in pos if v < SILENCE_RMS) / len(pos)
        sn = sum(1 for v in neg if v < SILENCE_RMS) / len(neg)
        print(f"무음 비율: pos {sp:.1%} / neg {sn:.1%}")
        print("-" * 56)
        if max(sp, sn) > 0.3:
            print("해석: 무음 비율 높음 -> 상당수 클립에 오디오 신호 부재. 쏠림의 데이터적 원인 가능.")
        elif abs(cohen) < 0.2:
            print("해석: pos/neg RMS 거의 겹침 -> 오디오 에너지만으론 판별 신호 약함(내용 차이는 별도).")
        else:
            print("해석: 오디오 신호 존재 + pos/neg 분포 차이 있음 -> 신호는 있음. 다음: ablation으로 모델이 쓰는지 확인.")


if __name__ == "__main__":
    main()