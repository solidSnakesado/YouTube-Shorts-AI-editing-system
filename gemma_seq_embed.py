# 47일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_seq_embed.py
#
# [수정 1회, 47일차] 재개(resume): 출력 npz 존재 시 해당 split 스킵(--overwrite로 강제 재추출).
#   사유: Colab 런타임 끊김 시 train(수 시간) 재실행 낭비 방지(head_embed 수정 2회와 동일 패턴).
#   변경 라인(본 파일 기준): L210~211(--overwrite 인자), L220~223(경로 선계산+스킵 분기).
#
# 목적: 시간축 보존 진단 1단계. C-1(평균 풀링)이 모든 모달 0.26~0.29 천장 -> 프레임 수
#   늘려도(8->30) 시각 0.21->0.26 미미 -> "평균 풀링이 시점 정보를 죽이는 게 천장 원인"
#   가설. 평균 풀링을 제거하고 30시점 시퀀스로 임베딩 -> 시계열 헤드(seq_train)가
#   "30초 중 어느 시점이 튀는가"를 보게 한다. 신호가 살아나면 평균 풀링이 범인 확정.
#
# 시간축 구조(probe 확인):
#   시각: get_image_features 출력 (30*264, 2560) -> [30프레임, 264패치, 2560]
#         -> 패치축 평균 -> [30, 2560] (프레임=시점, 1fps)
#   오디오: audio_tower 출력 (1, 750, 1536) -> 750스텝/30초 -> 30구간 묶음 평균
#         -> [30, 1536] (시각과 동일 30시점 정렬, 1초 단위)
#   결과: 시각[30,2560] + 오디오[30,1536] 각 저장. 시점별 concat은 헤드에서.
#
# npz 크기: 클립당 ~480KB(30x4096 f32) x 6058 ~ 2.9GB. 750스텝 원본은 70GB라 불가
#   -> 30구간 묶기 필수. 가변 프레임(29/30) 대응: 30 미만이면 마지막 시점 패딩+마스크.
#
# 출력: seq_cache_train.npz / seq_cache_eval.npz
#   Sv(N,30,2560) Sa(N,30,1536) M(N,30 유효시점 마스크) y(N,). 실패 클립 제외.
# 의존: gemma_collate, transformers, torch, numpy. 실행: python gemma_seq_embed.py [--limit 200]
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from gemma_collate import load_audio, load_images, parse_sample, sample_frames

BASE_MODEL = "unsloth/gemma-4-E4B-it"
T = 30                                          # 통일 시점 수(시각 1fps / 오디오 30구간)
PATCH_PER_FRAME = 264                           # 시각: 프레임당 토큰(7920/30, probe 확인)
_NUM_RE = re.compile(r"[01]?\.\d+|[01]")


def load_base(base_model: str):
    """베이스(bf16) + 프로세서. transformers 멀티모달 클래스(분리도/ C-1과 동일)."""
    import torch
    import transformers
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def _bin_mean(x, n_bins: int):
    """[L, D] -> [n_bins, D] 균등 구간 평균(L<n_bins면 그대로 반환, 패딩은 호출측)."""
    import torch

    L = x.shape[0]
    if L <= n_bins:
        return x
    edges = [round(i * L / n_bins) for i in range(n_bins + 1)]
    out = [x[edges[i]:max(edges[i] + 1, edges[i + 1])].mean(0) for i in range(n_bins)]
    return torch.stack(out, dim=0)


def embed_visual_seq(model, processor, parsed: dict, base_dir: Optional[str]):
    """프레임 시퀀스 -> [t, 2560] (t<=30, 프레임별 패치평균). 평균 풀링 안 함."""
    import torch

    frames = sample_frames(parsed["frame_paths"], 0)        # 전체(~30장)
    images = load_images(frames, base_dir)
    blocks = [{"type": "image"} for _ in frames]
    blocks.append({"type": "text", "text": "."})
    text = processor.apply_chat_template(
        [{"role": "user", "content": blocks}], tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[images], return_tensors="pt", padding=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    pv = inputs.get("pixel_values")
    if pv is None:
        return None
    nf = len(frames)
    with torch.no_grad():
        gif = getattr(model, "get_image_features", None)
        ipi = inputs.get("image_position_ids")
        out = (gif(pv, ipi) if ipi is not None else gif(pv))
        feats = getattr(out, "pooler_output", out)          # [nf*264, 2560] 기대
    if feats.ndim != 2 or feats.shape[0] % nf != 0:
        return None
    per = feats.shape[0] // nf
    seq = feats.view(nf, per, feats.shape[1]).float().mean(dim=1)   # [nf, 2560] 패치평균
    return seq.cpu()                                        # 시점=프레임 보존


def embed_audio_seq(model, processor, parsed: dict, base_dir: Optional[str]):
    """오디오 -> [t, 1536] (audio_tower 750스텝 -> 30구간 평균). 평균 풀링 안 함."""
    import torch

    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)
    inputs = processor(text=["."], audio=[wav], return_tensors="pt",
                       padding=True, truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    af = inputs.get("input_features")
    if af is None:
        return None
    mask = inputs.get("input_features_mask")
    at = getattr(model, "audio_tower", None) or \
        getattr(getattr(model, "model", None), "audio_tower", None)
    if at is None:
        return None
    with torch.no_grad():
        try:
            out = (at(af, mask) if mask is not None else at(af))
        except Exception:                           # noqa: BLE001
            out = at(af)
        feats = getattr(out, "last_hidden_state", out)      # [1, 750, 1536] 기대
    if feats.ndim == 3:
        feats = feats[0]
    seq = _bin_mean(feats.float(), T)                       # [<=T, 1536]
    return seq.cpu()


def _pad_to_T(seq, dim: int):
    """[t, dim] -> ([T, dim], valid_t). t<T면 마지막 시점 복제 패딩."""
    import torch

    t = seq.shape[0]
    if t >= T:
        return seq[:T], T
    pad = seq[-1:].repeat(T - t, 1)
    return torch.cat([seq, pad], dim=0), t


def parse_target(example: dict):
    try:
        txt = example["messages"][1]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    m = _NUM_RE.search(str(txt))
    return float(m.group()) if m else None


def load_rows(path: str, limit: int):
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if 0 < limit <= len(rows):
                break
    return rows


def build_cache(model, processor, rows, base_dir, tag: str) -> dict:
    import torch

    Sv_list, Sa_list, M_list, y_list = [], [], [], []
    skip = 0
    for i, row in enumerate(rows):
        y = parse_target(row)
        if y is None:
            skip += 1
            continue
        parsed = parse_sample(row)
        sv = embed_visual_seq(model, processor, parsed, base_dir)
        sa = embed_audio_seq(model, processor, parsed, base_dir)
        if sv is None or sa is None:
            skip += 1
            continue
        sv_p, vt = _pad_to_T(sv, sv.shape[1])
        sa_p, at_ = _pad_to_T(sa, sa.shape[1])
        valid = min(vt, at_)                        # 두 모달 공통 유효 시점
        if (not torch.all(torch.isfinite(sv_p)) or not torch.all(torch.isfinite(sa_p))):
            skip += 1
            continue
        mask = np.zeros(T, dtype=np.float32)
        mask[:valid] = 1.0
        Sv_list.append(sv_p.numpy().astype(np.float32))
        Sa_list.append(sa_p.numpy().astype(np.float32))
        M_list.append(mask)
        y_list.append(float(y))
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(rows)} (수집 {len(y_list)}, 스킵 {skip})")

    if not y_list:
        print(f"  WARN {tag}: 수집 0개")
        return {}
    Sv = np.stack(Sv_list); Sa = np.stack(Sa_list)
    M = np.stack(M_list); y = np.asarray(y_list, dtype=np.float32)
    print(f"  {tag} 완료: N={len(y)} | Sv{Sv.shape} Sa{Sa.shape} "
          f"유효시점평균={M.sum(1).mean():.1f}/{T} | y[std={y.std():.3f}] 스킵 {skip}")
    return {"Sv": Sv, "Sa": Sa, "M": M, "y": y}


def main() -> None:
    ap = argparse.ArgumentParser(description="시간축 보존 시퀀스 임베딩(30시점)")
    ap.add_argument("--train", default="datasets/gemma_audio/train_numeric.jsonl")
    ap.add_argument("--eval", default="datasets/gemma_audio/eval_numeric.jsonl")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--out-dir", default="datasets/gemma_audio/seq")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="47일차: 기존 npz 있어도 재추출(기본은 존재 시 스킵)")
    args = ap.parse_args()

    print(f"=== base 로드: {args.base_model} | 시퀀스 {T}시점 (평균풀링 제거) ===")
    model, processor = load_base(args.base_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, path in (("train", args.train), ("eval", args.eval)):
        out_path = out_dir / f"seq_cache_{split}.npz"           # 47일차: 경로 선계산
        if out_path.exists() and not args.overwrite:            # 47일차: 재개 스킵
            print(f"\n=== {split}: 기존 캐시 존재 -> 스킵 ({out_path}) ===")
            continue
        rows = load_rows(path, args.limit)
        print(f"\n=== {split} 시퀀스 추출: {len(rows)}행 ===")
        cache = build_cache(model, processor, rows, args.base_dir, split)
        if not cache:
            continue
        np.savez(out_path, **cache)
        print(f"  저장: {out_path}")

    print("\n완료. 다음: gemma_seq_train.py 로 시계열 헤드 학습/판정.")


if __name__ == "__main__":
    main()