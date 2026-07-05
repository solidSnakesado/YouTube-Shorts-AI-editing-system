# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_audio_separability.py
#
# 목적: (b) 입력 신호 검증 B단계. 시각 임베딩 분리도가 59.2%(약한 신호, 학습 어려움)로
#   확인됨 -> 오디오가 이를 보완하는지 측정. base 오디오 인코더(동결)로 pos/neg 오디오
#   임베딩 -> 분리도. 시각 스크립트와 동일 측정(로지스틱 5-fold CV)으로 직접 비교.
#
#   판별(시각 59.2% 대비):
#     - 오디오 분리도 시각보다 높음(예: 75%+) -> 오디오가 진짜 신호. 피벗 옳았고 모델이
#       오디오를 충분히 못 씀(인코더 학습 등 복귀 여지).
#     - 시각과 비슷/낮음(~59% 이하) -> 입력 전체(시각+오디오) 약함. (b) 완전 확정.
#       라벨 재정의/태스크 근본 재설계 필요.
#
# 오디오 임베딩: input_features를 오디오 인코더(audio_tower/get_audio_features 등 런타임 탐색)
#   에 통과 -> 시퀀스 특징 -> 평균 풀링. 생성 아님. base 로드는 ablation.load_base 재사용.
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py + gemma_audio_ablation.py
#   + gemma_visual_separability.py(report 재사용). sklearn(Colab 기본).
# 실행(Colab A100, 학습 셀 정지 후):
#   python gemma_audio_separability.py --n 60
from __future__ import annotations

import argparse
from typing import Optional

from gemma_audio_ablation import load_base, select
from gemma_collate import load_audio, parse_sample
from gemma_visual_separability import report


def _audio_features(model, inputs):
    """input_features -> 오디오 인코더 통과 특징 텐서. 여러 경로 런타임 탐색."""
    feats = inputs.get("input_features")
    if feats is None:
        return None
    mask = inputs.get("input_features_mask")
    # 경로1: get_audio_features(표준)
    gaf = getattr(model, "get_audio_features", None)
    if callable(gaf):
        try:
            out = (gaf(feats, mask) if mask is not None else gaf(feats))
            return getattr(out, "last_hidden_state", out)
        except Exception:                           # noqa: BLE001
            pass
    # 경로2: audio_tower 직접(모델 또는 model.model 하위)
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
    import numpy as np
    import torch

    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)
    # 오디오만 프로세싱(텍스트는 프로세서 요구로 더미)
    inputs = processor(text=["."], audio=[wav], return_tensors="pt",
                       padding=True, truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        feats = _audio_features(model, inputs)
    if feats is None:
        return None
    t = feats.float().cpu()
    return np.asarray(t.mean(dim=tuple(range(t.ndim - 1))))   # 마지막 차원 빼고 평균


def collect(model, processor, rows, base_dir, tag):
    """rows 오디오 임베딩 리스트(실패 건너뜀)."""
    import numpy as np

    vecs = []
    for i, row in enumerate(rows):
        parsed = parse_sample(row)
        v = embed_audio(model, processor, parsed, base_dir)
        if v is None or not np.all(np.isfinite(v)):
            print(f"  {tag}[{i}] 임베딩 실패(건너뜀)")
            continue
        vecs.append(v)
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser(description="오디오 임베딩 pos/neg 분리도(학습0, base 인코더)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--n", type=int, default=60, help="pos/neg 각 표본 수")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    pos_rows, neg_rows = select(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)}")
    print(f"=== base 로드: {args.base_model} (어댑터 없이) ===")
    model, processor = load_base(args.base_model)

    print("=== pos 오디오 임베딩 ===")
    pv = collect(model, processor, pos_rows, args.base_dir, "pos")
    print("=== neg 오디오 임베딩 ===")
    nv = collect(model, processor, neg_rows, args.base_dir, "neg")
    print("-" * 56)
    print("[오디오 분리도] (시각 59.2%와 비교)")
    report(pv, nv)


if __name__ == "__main__":
    main()