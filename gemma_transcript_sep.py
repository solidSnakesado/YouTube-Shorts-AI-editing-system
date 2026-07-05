# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_transcript_sep.py
#
# 목적: A+B 통합. 음성 스캔서 음성 클립 51.2%(pos55/neg47.5) 확인 -> 무음 게임플레이 아님,
#   음성 풍부 콘텐츠. Qwen은 transcript로 82% 달성했으나 Gemma는 transcript 빼고 오디오
#   원음만 줘서 59%. 음성 클립만 모아 transcript를 다시 넣으면 분리도가 82% 회복되는지 측정.
#
#   한 번의 전사로 4가지 분리도 동시 측정(전사가 지배적 비용이라 A/B 통합이 효율적):
#     1) transcript 단독 -> Qwen 82%와 1:1 비교(핵심). Qwen이 본 게 정확히 이것.
#     2) 시각 단독 -> 기존 59% 재확인(음성 클립 부분집합 기준)
#     3) 오디오 단독 -> 기존 59% 재확인
#     4) 셋 concat -> Gemma 전체 입력 상한
#
#   판별:
#     - transcript 단독 82% 근처 -> Gemma도 자막 주면 Qwen만큼 가능. 한계는 입력(자막 안 줌),
#       모델 아님. 무음만 본질적 한계. 결론: 학습에 transcript 추가.
#     - transcript 단독도 59% -> Gemma 텍스트 임베딩이 약함(모델 한계). 다른 길 필요.
#
# 흐름: 음성 클립만(Whisper 전사 비어있지 않은 것) -> transcript/프레임/오디오 임베딩 ->
#   각각 + concat 분리도(로지스틱 5-fold CV). report는 visual_separability 재사용.
#   텍스트 임베딩: base 언어모델에 transcript 통과 -> hidden state 평균(런타임 탐색).
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py + gemma_audio_ablation.py
#   + gemma_visual_separability.py(embed_frames, report) + gemma_audio_separability.py
#   (_audio_features) + gemma_voice_scan.py(transcribe, load_whisper). sklearn, faster-whisper.
# 실행(Colab A100 - Whisper+임베딩 함께):
#   python gemma_transcript_sep.py --n 80
from __future__ import annotations

import argparse

from gemma_audio_ablation import load_base, select
from gemma_audio_separability import _audio_features
from gemma_collate import load_audio, parse_sample
from gemma_visual_separability import embed_frames, report
from gemma_voice_scan import load_whisper, transcribe


def embed_text(model, processor, text: str):
    """transcript 텍스트를 언어모델에 통과 -> hidden state 평균 풀링(numpy)."""
    import numpy as np
    import torch

    tok = processor.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = tok["input_ids"].to(dev)
    mask = tok.get("attention_mask")
    mask = mask.to(dev) if mask is not None else None

    with torch.no_grad():
        # 언어모델 본체 런타임 탐색(get_input_embeddings -> 본체 forward)
        lm = getattr(model, "language_model", None) or \
            getattr(getattr(model, "model", None), "language_model", None) or \
            getattr(model, "model", None)
        if lm is None:
            return None
        try:
            out = lm(input_ids=ids, attention_mask=mask,
                     output_hidden_states=False)
        except Exception:                           # noqa: BLE001
            return None
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            return None
    # attention_mask로 유효 토큰만 평균
    h = hs.float().cpu()[0]                          # (seq, dim)
    if mask is not None:
        m = mask.cpu()[0].bool()
        h = h[m]
    return np.asarray(h.mean(dim=0))


def collect(whisper, model, processor, rows, base_dir, min_chars, tag):
    """음성 클립만 -> (transcript, 프레임, 오디오) 임베딩 3종 동시 수집."""
    import numpy as np

    t_vecs, v_vecs, a_vecs = [], [], []
    kept = 0
    for i, row in enumerate(rows):
        parsed = parse_sample(row)
        # 1) 전사 -> 음성 없으면 건너뜀
        try:
            wav, sr = load_audio(parsed["audio_path"], base_dir=base_dir)
            text = transcribe(whisper, wav, sr)
        except Exception:                           # noqa: BLE001
            continue
        if len(text) < min_chars:
            continue
        # 2) 세 임베딩
        tv = embed_text(model, processor, text)
        vv = embed_frames(model, processor, parsed, base_dir, 8)
        av_inputs = _audio_inputs(processor, wav, base_dir)
        av = _audio_embed(model, av_inputs)
        if any(x is None or not np.all(np.isfinite(x)) for x in (tv, vv, av)):
            continue
        t_vecs.append(tv)
        v_vecs.append(vv)
        a_vecs.append(av)
        kept += 1
        if (i + 1) % 20 == 0:
            print(f"  {tag} 진행 {i + 1}/{len(rows)} (유효 {kept})")
    print(f"  {tag} 음성+임베딩 유효: {kept}/{len(rows)}")
    return t_vecs, v_vecs, a_vecs


def _audio_inputs(processor, wav, base_dir):
    """오디오 임베딩용 입력 구성(audio_separability와 동일)."""
    import torch
    inputs = processor(text=["."], audio=[wav], return_tensors="pt",
                       padding=True, truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _audio_embed(model, inputs):
    """오디오 인코더 임베딩 -> 평균 풀링(numpy)."""
    import numpy as np
    import torch
    with torch.no_grad():
        feats = _audio_features(model, inputs)
    if feats is None:
        return None
    t = feats.float().cpu()
    return np.asarray(t.mean(dim=tuple(range(t.ndim - 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description="transcript/시각/오디오/합친 분리도 통합 측정")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--n", type=int, default=80, help="pos/neg 각 표본 수(음성 클립만 남음)")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--whisper", default="medium", help="medium 또는 large-v3-turbo")
    ap.add_argument("--min-chars", type=int, default=10)
    args = ap.parse_args()

    pos_rows, neg_rows = select(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)} (음성 클립만 측정에 사용)")
    print(f"=== Whisper 로드: {args.whisper} ===")
    whisper = load_whisper(args.whisper)
    print(f"=== base 로드: {args.base_model} ===")
    model, processor = load_base(args.base_model)

    print("=== pos 수집(전사+임베딩) ===")
    pt, pv, pa = collect(whisper, model, processor, pos_rows, args.base_dir,
                         args.min_chars, "pos")
    print("=== neg 수집(전사+임베딩) ===")
    nt, nv, na = collect(whisper, model, processor, neg_rows, args.base_dir,
                         args.min_chars, "neg")

    print("=" * 56)
    print("[1] transcript 단독 분리도 (★ Qwen 82%와 비교)")
    report(pt, nt)
    print("\n[2] 시각 단독 분리도 (기존 59% 재확인)")
    report(pv, nv)
    print("\n[3] 오디오 단독 분리도 (기존 59% 재확인)")
    report(pa, na)
    print("\n[4] 셋(transcript+시각+오디오) concat 분리도 (전체 상한)")
    _report_concat(pt, pv, pa, nt, nv, na)


def _report_concat(pt, pv, pa, nt, nv, na) -> None:
    """세 임베딩 이어붙여 분리도. 길이 맞는 것만 사용."""
    import numpy as np
    n_pos = min(len(pt), len(pv), len(pa))
    n_neg = min(len(nt), len(nv), len(na))
    if n_pos < 5 or n_neg < 5:
        print("표본 부족(각 5개 이상 필요)")
        return
    def cat(ts, vs, as_):
        out = []
        for t, v, a in zip(ts, vs, as_):
            out.append(np.concatenate([t, v, a]))
        return out
    report(cat(pt[:n_pos], pv[:n_pos], pa[:n_pos]),
           cat(nt[:n_neg], nv[:n_neg], na[:n_neg]))


if __name__ == "__main__":
    main()