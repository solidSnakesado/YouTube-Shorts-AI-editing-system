# 46일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_seq_train.py
# [수정 1회] torch.gelu -> F.gelu(L73). 사유: torch 2.11에 torch.gelu 미존재
#   (torch.nn.functional.gelu가 정식). build_head 내 import에 F 추가.
#
# 목적: 시간축 보존 2단계(최종 판정). gemma_seq_embed의 30시점 시퀀스(Sv/Sa) 위에
#   경량 시계열 헤드(self-attention) + attention 풀링 + 랭킹 손실로 학습 -> eval Spearman.
#   C-1(평균 풀링) 0.26~0.29가 천장이었는데, 시간축 살리면 오르는지로 "평균 풀링이 범인인가"
#   를 확정한다.
#
#   판정(C-1 concat 0.26 / aud 0.29 대비):
#     - Spearman 0.40+ -> 평균 풀링이 범인 확정. 신호 충분, 표현이 죽였던 것.
#       멀티모달 시계열 모델이 정답(영상 목적과 정합). 동결 인코더+작은 헤드로 충분.
#     - 0.30 정체 -> 평균 풀링도 범인 아님. 입력 신호 자체 한계 -> transcript(B) 또는
#       LLM+회귀헤드(길1, 더 큰 표현력) 검토.
#
# 헤드 구조: 시점별 [시각2560+오디오1536=4096] -> 입력투영 -> self-attention(시점 간 관계)
#   -> 시점별 스칼라 + attention 가중 풀링(어느 시점이 중요한지 학습; 단순 평균과 핵심 차이)
#   -> 클립 점수. 마스크로 패딩 시점 제외.
#
# 입력: seq_cache_train/eval.npz (Sv,Sa,M,y). --feat로 vis/aud/concat 선택.
# 의존: numpy, torch, scipy(폴백 내장). 실행(GPU 권장이나 CPU도 가능, 시퀀스라 C-1보다 무거움):
#   python gemma_seq_train.py
#   python gemma_seq_train.py --feat aud      # 오디오 시퀀스만(C-1 aud 0.29와 비교)
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
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


def load_seq(out_dir: Path, split: str, feat: str):
    """npz -> (X[N,T,D], M[N,T], y[N]). feat로 시각/오디오/concat."""
    d = np.load(out_dir / f"seq_cache_{split}.npz")
    sv, sa = d["Sv"].astype(np.float32), d["Sa"].astype(np.float32)
    if feat == "vis":
        X = sv
    elif feat == "aud":
        X = sa
    else:
        X = np.concatenate([sv, sa], axis=2)        # 시점별 결합 [N,T,4096]
    return X, d["M"].astype(np.float32), d["y"].astype(np.float32)


def build_head(in_dim: int, hidden: int, heads: int, dropout: float):
    """시계열 헤드: 투영 -> self-attention 1층 -> 시점 스칼라 + attention 풀링."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SeqHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.drop = nn.Dropout(dropout)
            self.score = nn.Linear(hidden, 1)       # 시점별 점수
            self.pool_w = nn.Linear(hidden, 1)      # attention 풀링 가중치

        def forward(self, x, m):                    # x[B,T,D] m[B,T]
            h = F.gelu(self.proj(x))
            kpm = (m < 0.5)                          # True=패딩(무시)
            a, _ = self.attn(h, h, h, key_padding_mask=kpm)
            h = self.norm(h + self.drop(a))         # 잔차
            sc = self.score(h).squeeze(-1)          # [B,T] 시점 점수
            w = self.pool_w(h).squeeze(-1)          # [B,T] 풀링 로짓
            w = w.masked_fill(kpm, float("-inf"))
            w = torch.softmax(w, dim=1)             # 유효 시점 분포
            clip = (sc * w).sum(dim=1)              # 가중 집계(학습된 풀링)
            return torch.sigmoid(clip)

    return SeqHead()


def pairwise_rank_loss(pred, y, margin: float):
    import torch

    p, yy = pred.view(-1), y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = yy.unsqueeze(1) - yy.unsqueeze(0)
    mask = (dy > 0).float()
    denom = mask.sum().clamp(min=1.0)
    return (torch.relu(margin - dp) * mask).sum() / denom


def evaluate(head, Xe, Me, ye, device) -> dict:
    import torch

    head.eval()
    with torch.no_grad():
        pe = head(torch.from_numpy(Xe).to(device),
                  torch.from_numpy(Me).to(device)).view(-1).cpu().numpy()
    return {"rho": spearman(pe, ye), "pred": pe, "std": float(pe.std()),
            "min": float(pe.min()), "max": float(pe.max()),
            "uniq": len({round(float(v), 3) for v in pe})}


def main() -> None:
    ap = argparse.ArgumentParser(description="시간축 보존 시계열 헤드 판정")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio/seq")
    ap.add_argument("--feat", default="concat", choices=["concat", "vis", "aud"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--mse-w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.cache_dir)

    Xtr, Mtr, ytr = load_seq(out_dir, "train", args.feat)
    Xte, Mte, yte = load_seq(out_dir, "eval", args.feat)
    # 시점별 표준화(전체 시점 통계)
    flat = Xtr.reshape(-1, Xtr.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    print(f"feat={args.feat} | train {Xtr.shape} eval {Xte.shape} | "
          f"ytr[std={ytr.std():.3f}] yte[std={yte.std():.3f}]")

    head = build_head(Xtr.shape[2], args.hidden, args.heads, args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Mtr_t = torch.from_numpy(Mtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    n = Xtr.shape[0]

    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s: s + args.batch]
            pred = head(Xtr_t[idx], Mtr_t[idx]).view(-1)
            yb = ytr_t[idx]
            loss = pairwise_rank_loss(pred, yb, args.margin) + args.mse_w * mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 10 == 0 or ep == 1:
            ev = evaluate(head, Xte, Mte, yte, device)
            print(f"[ep {ep:3d}] loss={tot / n:.4f} | eval Spearman={ev['rho']:.4f} "
                  f"pred[min={ev['min']:.3f} max={ev['max']:.3f} std={ev['std']:.3f} "
                  f"uniq={ev['uniq']}]")

    ev = evaluate(head, Xte, Mte, yte, device)
    print("\n" + "=" * 60)
    print(f"=== 시간축 보존 최종 판정 (feat={args.feat}) ===")
    print(f"eval Spearman 상관: {ev['rho']:.4f}  (C-1 평균풀링 concat 0.26 / aud 0.29 대비)")
    print(f"예측 분포: min={ev['min']:.4f} max={ev['max']:.4f} std={ev['std']:.4f} "
          f"고유값={ev['uniq']}개")
    buckets = [0] * 10
    for v in ev["pred"]:
        buckets[min(int(v * 10), 9)] += 1
    mx = max(buckets) or 1
    for i, c in enumerate(buckets):
        print(f"  [{i / 10:.1f}~{(i + 1) / 10:.1f}) {c:4d} {'#' * int(40 * c / mx)}")
    print("-" * 60)
    if ev["rho"] >= 0.38:
        print(f"해석: Spearman {ev['rho']:.2f} >> C-1(0.26~0.29) -> 평균 풀링이 범인 확정. "
              "시간축 살리니 신호 회복. 멀티모달 시계열 모델이 정답.")
    elif ev["rho"] <= 0.31:
        print(f"해석: Spearman {ev['rho']:.2f} ~ C-1 수준 -> 평균 풀링도 범인 아님. "
              "입력 신호 자체 한계. transcript(B) 또는 LLM+회귀헤드(길1) 검토.")
    else:
        print(f"해석: Spearman {ev['rho']:.2f} 소폭 상승 -> 시간축 일부 기여하나 약함. "
              "feat(vis/aud) 분해 + 헤드 용량 조정 검토.")


if __name__ == "__main__":
    main()