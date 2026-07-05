# 46일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_text_embed.py
#
# 목적: B-1 2단계(피벗 가설 판정). transcript의 하이라이트 예측력을 raw audio(C-1 0.29)와
#   비교. 피벗 가설="raw audio가 transcript보다 나음(효과음/BGM까지 잡으니)". 전사 텍스트를
#   Gemma 텍스트 인코더로 임베딩 -> C-1 시각/오디오 임베딩과 결합 -> 여러 feat로 Spearman.
#
#   판정(C-1 audio 0.29 대비):
#     - transcript-only < 0.29 -> 피벗 옳음. raw audio가 비언어까지 잡아 우세.
#     - transcript-only > 0.29 -> 피벗 재고. 말 의미가 더 강한 신호.
#     - audio+transcript >> 각각 -> 상보적. Gemma 멀티모달로 둘 다 써야(피벗의 진짜 정당화).
#
# 텍스트 임베딩: 전사 텍스트를 토크나이즈 -> model() last_hidden_state 평균(시각/오디오의
#   동결 인코더 임베딩과 같은 철학). 빈 전사(말 없는 구간)는 빈 텍스트 임베딩 -> transcript
#   신호의 실제 한계를 반영(인위적 보정 안 함).
# 정렬: transcript_{split}.jsonl의 idx로 C-1 npz 앞 N개와 매칭(전사가 같은 jsonl 순서).
#
# 입력: transcript_train/eval.jsonl + head_cache_train/eval.npz(C-1) + eval_numeric(타깃 검증).
# 출력: 콘솔 판정(없으면 npz 저장 옵션). 의존: transformers, torch, numpy, scipy(폴백).
# 실행: python gemma_text_embed.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(a, b) -> float:
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r == r else 0.0
    except Exception:                               # noqa: BLE001
        ar = np.argsort(np.argsort(a)).astype(np.float64)
        br = np.argsort(np.argsort(b)).astype(np.float64)
        ar -= ar.mean(); br -= br.mean()
        d = float(np.sqrt((ar ** 2).sum() * (br ** 2).sum())) or 1e-9
        return float((ar * br).sum() / d)


def load_base(base_model: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def embed_texts(model, processor, texts, batch: int = 16):
    """텍스트 리스트 -> [N, D] last_hidden_state 평균(attention mask 가중)."""
    import torch

    tok = processor.tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for s in range(0, len(texts), batch):
        chunk = [t if t.strip() else "." for t in texts[s: s + batch]]  # 빈텍스트 더미
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(dev)
        with torch.no_grad():
            lm = getattr(model, "language_model", None) or \
                getattr(getattr(model, "model", None), "language_model", None) or \
                getattr(model, "model", model)
            res = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                     output_hidden_states=False)
            hs = getattr(res, "last_hidden_state", None)
            if hs is None:
                hs = res[0]
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs.float() * m).sum(1) / m.sum(1).clamp(min=1.0)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def load_transcripts(path: str):
    """transcript jsonl -> (idx 리스트, text 리스트). idx 순서 보존."""
    idxs, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idxs.append(int(r["idx"]))
            texts.append(r.get("text", "") or "")
    return idxs, texts


def build_head(in_dim, hidden, dropout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1), nn.Sigmoid(),
    )


def rank_loss(pred, y, margin):
    import torch
    p, yy = pred.view(-1), y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = yy.unsqueeze(1) - yy.unsqueeze(0)
    mask = (dy > 0).float()
    return (torch.relu(margin - dp) * mask).sum() / mask.sum().clamp(min=1.0)


def train_eval(Xtr, ytr, Xte, yte, name, epochs=60, hidden=256):
    """헤드 학습 -> eval Spearman. (rho, pred std) 반환."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    head = build_head(Xtr.shape[1], hidden, 0.1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    n = Xtr.shape[0]
    for _ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 256):
            idx = perm[s: s + 256]
            pred = head(Xt[idx]).view(-1)
            loss = rank_loss(pred, yt[idx], 0.1) + 0.3 * mse(pred, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pe = head(torch.from_numpy(Xte).to(dev)).view(-1).cpu().numpy()
    rho = spearman(pe, yte)
    print(f"  [{name:<22}] Spearman={rho:.4f}  pred_std={pe.std():.3f}")
    return rho


def main() -> None:
    ap = argparse.ArgumentParser(description="B-1 2단계: transcript 예측력 vs raw audio")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    args = ap.parse_args()

    cd = Path(args.cache_dir)
    # 1) C-1 임베딩 로드
    dtr = np.load(cd / "head_cache_train.npz")
    dte = np.load(cd / "head_cache_eval.npz")
    # 2) transcript 로드(앞 N개) + C-1 npz 앞 N개로 정렬
    itr, ttr = load_transcripts(cd / "transcript_train.jsonl")
    ite, tte = load_transcripts(cd / "transcript_eval.jsonl")
    ntr, nte = len(itr), len(ite)
    assert itr == list(range(ntr)) and ite == list(range(nte)), "idx 비연속 -> 정렬 불가"
    print(f"샘플: train {ntr} / eval {nte} (C-1 npz 앞 N개와 정렬)")

    Xv_tr, Xa_tr = dtr["X_vis"][:ntr], dtr["X_aud"][:ntr]
    Xv_te, Xa_te = dte["X_vis"][:nte], dte["X_aud"][:nte]
    ytr, yte = dtr["y"][:ntr].astype(np.float32), dte["y"][:nte].astype(np.float32)

    # 3) 텍스트 임베딩
    print("=== base 로드 + transcript 임베딩 ===")
    model, processor = load_base(args.base_model)
    Xt_tr = embed_texts(model, processor, ttr)
    Xt_te = embed_texts(model, processor, tte)
    print(f"  텍스트 임베딩: train {Xt_tr.shape} eval {Xt_te.shape}")

    # 4) feat 조합별 Spearman
    print("\n" + "=" * 60)
    print(f"=== B-1 판정: transcript 예측력 vs raw audio (n_eval={nte}) ===")
    print(f"  (C-1 전체 audio 0.29 / 단 여기선 샘플 {nte}개라 값 다를 수 있음)")
    r_aud = train_eval(Xa_tr, ytr, Xa_te, yte, "audio only")
    r_txt = train_eval(Xt_tr, ytr, Xt_te, yte, "transcript only")
    r_vis = train_eval(Xv_tr, ytr, Xv_te, yte, "visual only")
    at = np.concatenate([Xa_tr, Xt_tr], 1), np.concatenate([Xa_te, Xt_te], 1)
    r_at = train_eval(at[0], ytr, at[1], yte, "audio+transcript")
    allf = (np.concatenate([Xv_tr, Xa_tr, Xt_tr], 1),
            np.concatenate([Xv_te, Xa_te, Xt_te], 1))
    r_all = train_eval(allf[0], ytr, allf[1], yte, "visual+audio+transcript")

    print("-" * 60)
    best_uni = max(r_aud, r_txt, r_vis)
    if r_txt > r_aud + 0.03:
        print(f"판정: transcript({r_txt:.2f}) > audio({r_aud:.2f}) -> 피벗 재고. "
              "말 의미가 더 강한 신호. transcript 복귀/추가 검토.")
    elif r_aud > r_txt + 0.03:
        print(f"판정: audio({r_aud:.2f}) > transcript({r_txt:.2f}) -> 피벗 옳음. "
              "raw audio가 비언어(효과음/BGM)까지 잡아 우세.")
    else:
        print(f"판정: audio({r_aud:.2f}) ~ transcript({r_txt:.2f}) 비슷.")
    if r_at > best_uni + 0.05 or r_all > best_uni + 0.05:
        print(f"  + 결합 이득: audio+txt={r_at:.2f}, all={r_all:.2f} >> 단일 최고 "
              f"{best_uni:.2f} -> 상보적. 멀티모달로 둘 다(Gemma 정당화).")
    else:
        print(f"  결합 이득 미미(audio+txt={r_at:.2f}, all={r_all:.2f}) -> "
              "모달 추가 효과 제한적.")


if __name__ == "__main__":
    main()