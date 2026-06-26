# 43일차 신규: Gemma GGUF LoRA 스케일 스윕 진단 (collapse vs 변환 미적용 판별)
# 배치 경로: 저장소 루트 ~/project/yt_shorts_ai/gemma_scale_sweep.py (독립 도구, 신규)
#
# 배경: greedy 8프레임에서 긍정(gt 0.9551)·부정(gt [])이 모두 동일하게 0.8229 출력 ->
#       모델이 입력 무관 상수만 뱉음(붕괴 의심). 두 가설을 구별한다:
#   H1 학습 붕괴 : 어댑터 자체가 입력 무시 -> 재학습 필요
#   H2 변환 미적용: GGUF 변환에서 LoRA 실효 스케일이 낮음 -> 스케일 올리면 구별 회복
# 방법: 스케일 [무LoRA, 2.0, 3.0, 4.0] x 샘플 {pos, neg} 행렬로 hook_score 표 생성.
#   - 어느 스케일에서 pos>neg(구별 양수)면 H2(변환 미적용 -> 스케일/머지로 해결)
#   - 모든 스케일에서 pos≈neg(구별≈0)면 H1(학습 붕괴 -> 재학습)
#   - 무LoRA(베이스)도 0.82 근처면 LoRA 사실상 미작동
#
# 사용: python3 gemma_scale_sweep.py
#       python3 gemma_scale_sweep.py --scales 2.0,2.5,3.0,3.5
# 주의: 매 호출마다 모델 재로딩 -> 8회 약 2~3분 소요(정상).
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gemma_collate import parse_sample, sample_frames
from gemma_mtmd_probe import find_row, BIN, MODEL, MMPROJ, LORA, N_PREDICT
from gemma_inference import parse_hook_score

DATA = "datasets/gemma_audio/all.jsonl"


def build_argv(frames, audio, prompt, scale, base_dir=None):
    """subprocess용 argv 리스트. scale=None 이면 LoRA 미적용(베이스)."""
    def media(p):
        return str(Path(base_dir) / p) if base_dir else p

    argv = [BIN, "-m", MODEL, "--mmproj", MMPROJ]
    if scale is not None:
        argv += ["--lora-scaled", f"{LORA}:{scale}"]
    argv += ["-ngl", "99", "--jinja", "--temp", "0"]
    argv += ["--image", ",".join(media(f) for f in frames)]
    argv += ["--audio", media(audio)]
    argv += ["-p", prompt, "-n", str(N_PREDICT)]
    return argv


def run_one(frames, audio, prompt, scale, base_dir=None):
    """1회 추론 -> hook_score. []출력은 0.0, 파싱실패/타임아웃은 None."""
    argv = build_argv(frames, audio, prompt, scale, base_dir)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return None
    score = parse_hook_score(proc.stdout + "\n" + proc.stderr)
    return None if score == -1.0 else score   # -1.0(파싱실패)은 None 으로


def pick(jsonl, kind, n_frames):
    """pos/neg 샘플 1개 -> (frames, audio, prompt, target, sid)."""
    _ln, row, _m = find_row(jsonl, None, kind)
    p = parse_sample(row)
    frames = sample_frames(p["frame_paths"], n_frames)
    sid = p["audio_path"].rsplit("/", 1)[-1]
    return frames, p["audio_path"], p["instruction"], p["target"], sid


def _fmt(score):
    """점수 표시: None->ERR, 0.0->[](부정정답), 그외 소수4자리."""
    if score is None:
        return "ERR"
    if score == 0.0:
        return "[](0.0)"
    return f"{score:.4f}"


def main():
    ap = argparse.ArgumentParser(description="Gemma GGUF LoRA 스케일 스윕 진단")
    ap.add_argument("--jsonl", default=DATA)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--scales", default="none,2.0,3.0,4.0",
                    help="콤마구분 스케일 목록. 'none'=LoRA 미적용(베이스)")
    args = ap.parse_args()

    pos = pick(args.jsonl, "pos", args.n_frames)
    neg = pick(args.jsonl, "neg", args.n_frames)
    print(f"pos 샘플: {pos[4]} (정답 {pos[3]})")
    print(f"neg 샘플: {neg[4]} (정답 {neg[3]})")
    scales = [None if s.strip().lower() == "none" else s.strip()
              for s in args.scales.split(",")]

    print("\n{:<8} {:>12} {:>12} {:>10}".format("scale", "pos", "neg", "구별(p-n)"))
    print("-" * 46)
    for sc in scales:
        ps = run_one(*pos[:3], sc, args.base_dir)
        ns = run_one(*neg[:3], sc, args.base_dir)
        if ps is not None and ns is not None:
            gap = f"{ps - ns:+.4f}"
        else:
            gap = "n/a"
        label = "none" if sc is None else str(sc)
        print("{:<8} {:>12} {:>12} {:>10}".format(label, _fmt(ps), _fmt(ns), gap))
    print("-" * 46)
    print("판독:")
    print("  어느 스케일에서 pos>neg(구별 양수↑) -> H2(변환 미적용, 스케일/머지로 해결)")
    print("  모든 스케일에서 pos≈neg(구별≈0)     -> H1(학습 붕괴, 재학습 필요)")
    print("  neg 가 [](0.0) -> 부정을 빈 리스트로 맞춘 것(정답 방향)")


if __name__ == "__main__":
    main()