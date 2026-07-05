# 47일차 수정(2회) | 배치: ~/project/yt_shorts_ai/gemma_head_embed.py
#
# [수정 2회, 47일차] 재개(resume) 기능: 출력 npz가 이미 존재하는 split은 건너뜀.
#   사유: Colab 런타임 끊김 시 train(2시간+)을 재실행하는 낭비 방지. train npz가
#   Drive에 저장 완료된 상태 -> eval만 이어서 추출 가능하게.
#   변경 라인(본 파일 기준): L216~217(--overwrite 인자), L228~231(경로 선계산+스킵 분기).
#
# [수정 1회] MAX_FRAMES 상수 -> --max-frames 인자(기본 0=전체). 사유: 클립 30초인데
#   8프레임(0.27fps)이 시각 정보를 과소 샘플링 -> C-1 시각 Spearman 0.21 저평가 의심.
#   디스크에 클립당 ~30프레임(1fps) 존재 확인 -> 0(전체)으로 시각 신호 재측정.
#   인코더만 통과(LLM 디코더 없음)라 비전토큰 max_length 제약 무관 -> 전체 안전.
#   변경: embed_frames에 max_frames 인자, build_cache 전달, main --max-frames 추가.
#
# 목적: C-1 1단계. (a) LLM 회귀가 round8까지 0.111 상수 붕괴 확정 -> "거대 vocab CE로
#   약한 신호를 회귀한 출력 형식"이 원인인지, "신호 자체가 약한지"를 가르는 진단.
#   동결 Gemma 인코더(분리도 59% 측정에 쓴 그 인코더)로 시각+오디오 임베딩을 뽑아
#   (임베딩, 연속 타깃 hook_score) 쌍을 npz로 캐시. 2단계(gemma_head_train)가 이 위에
#   작은 회귀 헤드 + 랭킹 손실로 학습 -> 점수가 퍼지고 타깃과 상관나는지 판정.
#
# 독립 재작성 사유: 분리도 스크립트 의존 체인(audio_ablation/visual_sep/audio_sep/
#   collapse_check)이 Colab Drive에 없어, 실제 쓰는 함수 3개(load_base/embed_frames/
#   embed_audio)를 본 파일에 내장. Colab엔 본 파일 + gemma_collate.py만 있으면 됨.
#   인코더는 transformers AutoModelForImageTextToText로 로드(분리도 측정과 동일 -> 같은
#   임베딩 -> 59% 천장과 일관). unsloth 불필요.
#
# 입력: train_numeric/eval_numeric.jsonl (target=순수 숫자 "0.73", messages 구조 동일).
# 출력: head_cache_train.npz / head_cache_eval.npz (X_vis, X_aud, X=concat, y). 실패 클립 제외.
# 의존: gemma_collate(load_images/load_audio/parse_sample/sample_frames), transformers, torch, numpy.
# 실행: python gemma_head_embed.py [--limit 200]
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from gemma_collate import load_audio, load_images, parse_sample, sample_frames

BASE_MODEL = "unsloth/gemma-4-E4B-it"           # bf16 원본(분리도 측정과 동일)
DEFAULT_MAX_FRAMES = 0                          # 0=전체 프레임 사용(30초 클립의 ~30장 전부)
_NUM_RE = re.compile(r"[01]?\.\d+|[01]")


def load_base(base_model: str):
    """베이스(bf16, 어댑터 없이) + 프로세서 로드. transformers 멀티모달 클래스."""
    import torch
    import transformers
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def embed_frames(model, processor, parsed: dict, base_dir: Optional[str],
                 max_frames: int):
    """프레임들을 비전 인코더로 임베딩 -> 평균 풀링 벡터(numpy). 생성 아님."""
    import torch

    frames = sample_frames(parsed["frame_paths"], max_frames)
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
    with torch.no_grad():
        feats = None
        gif = getattr(model, "get_image_features", None)
        if callable(gif):
            try:
                ipi = inputs.get("image_position_ids")
                out = (gif(pv, ipi) if ipi is not None else gif(pv))
                feats = getattr(out, "pooler_output", out)
            except Exception:                       # noqa: BLE001
                feats = None
        if feats is None:
            vt = getattr(model, "vision_tower", None) or \
                getattr(getattr(model, "model", None), "vision_tower", None)
            if vt is None:
                return None
            out = vt(pv)
            feats = getattr(out, "pooler_output", None)
            if feats is None:
                feats = getattr(out, "last_hidden_state", out)
    t = feats.float().cpu()
    return np.asarray(t.mean(dim=tuple(range(t.ndim - 1))))


def _audio_features(model, inputs):
    """input_features -> 오디오 인코더 통과 특징 텐서. 여러 경로 런타임 탐색."""
    feats = inputs.get("input_features")
    if feats is None:
        return None
    mask = inputs.get("input_features_mask")
    gaf = getattr(model, "get_audio_features", None)
    if callable(gaf):
        try:
            out = (gaf(feats, mask) if mask is not None else gaf(feats))
            return getattr(out, "last_hidden_state", out)
        except Exception:                           # noqa: BLE001
            pass
    at = getattr(model, "audio_tower", None) or \
        getattr(getattr(model, "model", None), "audio_tower", None)
    if at is None:
        return None
    try:
        out = (at(feats, mask) if mask is not None else at(feats))
    except Exception:                               # noqa: BLE001
        out = at(feats)
    return getattr(out, "last_hidden_state", out)


def embed_audio(model, processor, parsed: dict, base_dir: Optional[str]):
    """오디오를 오디오 인코더로 임베딩 -> 평균 풀링 벡터(numpy). 생성 아님."""
    import torch

    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)
    inputs = processor(text=["."], audio=[wav], return_tensors="pt",
                       padding=True, truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        feats = _audio_features(model, inputs)
    if feats is None:
        return None
    t = feats.float().cpu()
    return np.asarray(t.mean(dim=tuple(range(t.ndim - 1))))


def parse_target(example: dict):
    """assistant target 텍스트 -> float. numeric 형식("0.73") 직접 파싱, 실패 None."""
    try:
        txt = example["messages"][1]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    m = _NUM_RE.search(str(txt))
    return float(m.group()) if m else None


def load_rows(path: str, limit: int):
    """jsonl -> 행 리스트(limit>0이면 앞 limit개만)."""
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


def build_cache(model, processor, rows, base_dir, max_frames: int, tag: str) -> dict:
    """각 행 -> 시각/오디오 임베딩 + 타깃. 어느 하나라도 실패하면 그 클립 제외."""
    vis_list: list[np.ndarray] = []
    aud_list: list[np.ndarray] = []
    y_list: list[float] = []
    skip = 0

    for i, row in enumerate(rows):
        y = parse_target(row)
        if y is None:
            skip += 1
            continue
        parsed = parse_sample(row)
        v = embed_frames(model, processor, parsed, base_dir, max_frames)
        a = embed_audio(model, processor, parsed, base_dir)
        if (v is None or a is None
                or not np.all(np.isfinite(v)) or not np.all(np.isfinite(a))):
            skip += 1
            continue
        vis_list.append(np.asarray(v, dtype=np.float32).ravel())
        aud_list.append(np.asarray(a, dtype=np.float32).ravel())
        y_list.append(float(y))
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(rows)} 처리 (수집 {len(y_list)}, 스킵 {skip})")

    if not y_list:
        print(f"  WARN {tag}: 수집 0개")
        return {}

    X_vis = np.vstack(vis_list)
    X_aud = np.vstack(aud_list)
    X = np.concatenate([X_vis, X_aud], axis=1)
    y = np.asarray(y_list, dtype=np.float32)
    print(f"  {tag} 완료: N={len(y)} | vis={X_vis.shape[1]} aud={X_aud.shape[1]} "
          f"concat={X.shape[1]} | y[min={y.min():.3f} max={y.max():.3f} "
          f"mean={y.mean():.3f} std={y.std():.3f}] | 스킵 {skip}")
    return {"X_vis": X_vis, "X_aud": X_aud, "X": X, "y": y}


def main() -> None:
    ap = argparse.ArgumentParser(description="C-1 1단계: 동결 인코더 임베딩 + 타깃 캐시")
    ap.add_argument("--train", default="datasets/gemma_audio/train_numeric.jsonl")
    ap.add_argument("--eval", default="datasets/gemma_audio/eval_numeric.jsonl")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-dir", default=None, help="미디어 상대경로 기준(보통 None=cwd)")
    ap.add_argument("--out-dir", default="datasets/gemma_audio", help="npz 캐시 출력 디렉토리")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help="클립당 프레임(0=전체~30장, 8=기존). 시각 샘플링 해상도")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(스모크용, 0=전체)")
    ap.add_argument("--overwrite", action="store_true",
                    help="47일차: 기존 npz 있어도 재추출(기본은 존재 시 스킵)")
    args = ap.parse_args()

    print(f"=== base 로드: {args.base_model} (어댑터 없이, 동결 인코더) "
          f"| max_frames={args.max_frames if args.max_frames > 0 else '전체'} ===")
    model, processor = load_base(args.base_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, path in (("train", args.train), ("eval", args.eval)):
        out_path = out_dir / f"head_cache_{split}.npz"          # 47일차: 경로 선계산
        if out_path.exists() and not args.overwrite:            # 47일차: 재개 스킵
            print(f"\n=== {split}: 기존 캐시 존재 -> 스킵 ({out_path}) ===")
            continue
        rows = load_rows(path, args.limit)
        print(f"\n=== {split} 임베딩 추출: {len(rows)}행 ({path}) ===")
        cache = build_cache(model, processor, rows, args.base_dir, args.max_frames, split)
        if not cache:
            continue
        np.savez(out_path, **cache)
        print(f"  저장: {out_path}")

    print("\n완료. 다음: gemma_head_train.py 로 회귀 헤드 + 랭킹 손실 학습/판정.")


if __name__ == "__main__":
    main()