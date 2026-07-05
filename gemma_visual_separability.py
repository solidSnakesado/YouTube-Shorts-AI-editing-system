# 45일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_visual_separability.py
#
# [수정 1회] 임베딩 입력 구성 수정: "<image>" 더미텍스트 -> apply_chat_template로 image
#   블록 렌더(Gemma4 실토큰 <|image|> 매칭). 사유: "<image>"는 0토큰 카운트 -> ValueError.
#   변경 라인 아래 전달 메시지 참조.
#
# 목적: (b) 입력 신호 검증 - 핵심. 회귀5+분류2 붕괴, ablation서 오디오는 반영되나
#   pos/neg 변별 안 됨 확인 -> "프레임(시각)에 라벨을 예측할 신호가 있는가"를 학습 없이 측정.
#   base 비전 인코더(동결, ablation서 작동 확인)로 pos/neg 프레임 임베딩 -> 분리도.
#
#   판별:
#     - 임베딩만으로 pos/neg 선형 분류 정확도 높음(>70%) -> 시각 신호 있음.
#       입력엔 신호 있는데 LoRA가 못 잡음 -> 학습 방식 문제(증강/구조).
#     - 정확도 ~50%(랜덤) -> 시각에 라벨 신호 없음 -> (b) 확정. 라벨/태스크 재설계 필요.
#   base 인코더는 학습 무관 "프레임이 무엇인지" 일반표현 -> 라벨이 시각적으로 예측 가능한지의
#   오염 없는 척도(LoRA 학습 성패와 독립).
#
# 측정(학습 없음, 임베딩 추출만 GPU):
#   1) 효과크기: pos/neg 임베딩 중심 거리 / 클래스내 평균분산(분포 분리)
#   2) 로지스틱 5-fold CV 정확도: 임베딩만으로 pos/neg 분류 상한(주지표)
#
# 임베딩: 모델의 get_image_features(있으면) 또는 vision_tower로 프레임별 특징 -> 평균 풀링.
#   생성(generate) 아님 - forward 일부만. base 로드는 gemma_audio_ablation.load_base 재사용.
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py + gemma_audio_ablation.py.
#   추가: scikit-learn(Colab 기본 설치됨).
# 실행(Colab A100, 학습 셀 정지 후):
#   python gemma_visual_separability.py --n 60
#   python gemma_visual_separability.py --n 60 --max-frames 8
from __future__ import annotations

import argparse
from typing import Optional

from gemma_audio_ablation import load_base, select
from gemma_collate import load_images, parse_sample, sample_frames


def embed_frames(model, processor, parsed: dict, base_dir: Optional[str],
                 max_frames: int):
    """프레임들을 비전 인코더로 임베딩 -> 평균 풀링 벡터(numpy). 생성 아님."""
    import numpy as np
    import torch

    frames = sample_frames(parsed["frame_paths"], max_frames)
    images = load_images(frames, base_dir)
    # collate와 동일하게 image 블록을 apply_chat_template로 렌더(토큰 <|image|> 정확 매칭)
    blocks = [{"type": "image"} for _ in frames]
    blocks.append({"type": "text", "text": "."})       # 프로세서가 텍스트 요구(임베딩엔 무의미)
    text = processor.apply_chat_template(
        [{"role": "user", "content": blocks}], tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[images],
                       return_tensors="pt", padding=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    pv = inputs.get("pixel_values")
    if pv is None:
        return None
    with torch.no_grad():
        feats = None
        # 경로1: get_image_features(가장 표준)
        gif = getattr(model, "get_image_features", None)
        if callable(gif):
            try:
                ipi = inputs.get("image_position_ids")
                out = (gif(pv, ipi) if ipi is not None else gif(pv))
                feats = getattr(out, "pooler_output", out)
            except Exception:                       # noqa: BLE001
                feats = None
        # 경로2: vision_tower 직접
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
    return np.asarray(t.mean(dim=tuple(range(t.ndim - 1))))   # 마지막 차원 빼고 평균


def collect_embeddings(model, processor, rows, base_dir, max_frames, tag):
    """rows 임베딩 리스트(실패 건너뜀). 진행 출력."""
    import numpy as np

    vecs = []
    for i, row in enumerate(rows):
        parsed = parse_sample(row)
        v = embed_frames(model, processor, parsed, base_dir, max_frames)
        if v is None or not np.all(np.isfinite(v)):
            print(f"  {tag}[{i}] 임베딩 실패(건너뜀)")
            continue
        vecs.append(v)
    return vecs


def report(pos_vecs, neg_vecs) -> None:
    """효과크기 + 로지스틱 5-fold CV 정확도."""
    import numpy as np

    if len(pos_vecs) < 5 or len(neg_vecs) < 5:
        print("표본 부족(각 5개 이상 필요)")
        return
    dim = min(len(v) for v in pos_vecs + neg_vecs)
    P = np.stack([v[:dim] for v in pos_vecs])
    N = np.stack([v[:dim] for v in neg_vecs])
    print(f"임베딩 차원: {dim} | pos {len(P)} / neg {len(N)}")

    # 1) 효과크기: 중심 거리 / 클래스내 평균 표준편차
    cp, cn = P.mean(0), N.mean(0)
    dist = float(np.linalg.norm(cp - cn))
    spread = float((P.std(0).mean() + N.std(0).mean()) / 2) or 1e-9
    print(f"중심 거리={dist:.3f}  클래스내 평균std={spread:.3f}  "
          f"비율(거리/std)={dist / spread:.3f}")

    # 2) 로지스틱 5-fold CV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([P, N])
    y = np.array([1] * len(P) + [0] * len(N))
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    k = min(5, len(P), len(N))
    scores = cross_val_score(clf, Xs, y, cv=k, scoring="accuracy")
    acc = float(scores.mean())
    print(f"로지스틱 {k}-fold CV 정확도: {acc:.1%} (fold별 "
          f"{', '.join(f'{s:.0%}' for s in scores)})")
    print("-" * 56)
    base = max(len(P), len(N)) / (len(P) + len(N))    # 다수클래스 기준선
    if acc >= 0.70:
        print(f"해석: 시각 임베딩으로 pos/neg 분류 {acc:.1%}(기준선 {base:.1%}) -> "
              "프레임에 라벨 신호 있음. 입력엔 신호 존재, LoRA가 못 잡음(학습 방식 문제).")
    elif acc <= base + 0.05:
        print(f"해석: {acc:.1%} ~ 기준선({base:.1%}) -> 시각에 라벨 신호 거의 없음. "
              "(b) 확정. 라벨/태스크 재설계 필요. 다음: 오디오 임베딩(B)도 확인.")
    else:
        print(f"해석: {acc:.1%}(기준선 {base:.1%}) 약한 신호 -> 부분적. "
              "증강으로 끌어올릴 여지 있으나 약함. B(오디오) 확인 권장.")


def main() -> None:
    ap = argparse.ArgumentParser(description="프레임 임베딩 pos/neg 분리도(학습0, base 인코더)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--n", type=int, default=60, help="pos/neg 각 표본 수(많을수록 안정)")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--max-frames", type=int, default=8, help="클립당 프레임(학습과 동일 8)")
    args = ap.parse_args()

    pos_rows, neg_rows = select(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)}")
    print(f"=== base 로드: {args.base_model} (어댑터 없이) ===")
    model, processor = load_base(args.base_model)

    print("=== pos 임베딩 ===")
    pv = collect_embeddings(model, processor, pos_rows, args.base_dir,
                            args.max_frames, "pos")
    print("=== neg 임베딩 ===")
    nv = collect_embeddings(model, processor, neg_rows, args.base_dir,
                            args.max_frames, "neg")
    print("-" * 56)
    report(pv, nv)


if __name__ == "__main__":
    main()