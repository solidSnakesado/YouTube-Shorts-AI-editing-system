# 46일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_head_train.py
#
# 목적: C-1 2단계(최종 판정). gemma_head_embed가 만든 동결 임베딩(npz) 위에 작은 회귀
#   헤드(MLP) + 랭킹 손실(pairwise margin)로 학습 -> eval에서 예측이 타깃과 상관나는지,
#   분포가 퍼지는지로 "(a) 0.111 붕괴가 LLM 출력 형식 탓인가 vs 신호 자체가 약한가"를 가른다.
#
#   랭킹 손실 핵심: 배치 내 (높은 타깃 i, 낮은 타깃 j) 쌍에서 pred_i > pred_j 강제(margin).
#   상수 출력은 모든 쌍을 틀리므로 강하게 벌점 -> LLM 거대 vocab CE엔 없던 "퍼짐" 압력.
#   MSE를 약하게 병행(--mse-w)해 출력 스케일을 0~1로 고정(랭킹만이면 스케일 자유).
#
#   판정(eval 기준):
#     - Spearman >= 0.3 + 예측 std 큼(분포 퍼짐) -> 같은 동결 임베딩인데 헤드+랭킹으로
#       점수 살아남 -> 붕괴 원인은 신호 아니라 LLM 출력 형식. 전용 헤드 방향 정당화.
#     - Spearman ~0 + 예측 뭉침(0.11류) -> 동결 임베딩 자체에 신호 부족. 아키텍처로 못
#       고침 -> transcript 입력 추가가 유일한 길. 무음 게임플레이는 본질적 한계.
#
# 입력: head_cache_train.npz / head_cache_eval.npz (X, X_vis, X_aud, y). --feat로 입력 선택.
# 의존: numpy, torch, scipy(spearman; 없으면 자체 순위상관 폴백). 학습 데이터/모델 무관.
# 실행(로컬 또는 Colab; npz만 있으면 GPU 불필요):
#   python gemma_head_train.py
#   python gemma_head_train.py --feat vis     # 시각만 / aud 오디오만 / concat 둘다(기본)
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관(scipy 있으면 사용, 없으면 순위 변환 후 피어슨)."""
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r == r else 0.0          # NaN 방어
    except Exception:                               # noqa: BLE001
        ar = np.argsort(np.argsort(a)).astype(np.float64)
        br = np.argsort(np.argsort(b)).astype(np.float64)
        ar -= ar.mean(); br -= br.mean()
        d = float(np.sqrt((ar ** 2).sum() * (br ** 2).sum())) or 1e-9
        return float((ar * br).sum() / d)


def load_split(out_dir: Path, split: str, feat: str):
    """npz -> (X[feat], y). feat in {concat, vis, aud}."""
    d = np.load(out_dir / f"head_cache_{split}.npz")
    key = {"concat": "X", "vis": "X_vis", "aud": "X_aud"}[feat]
    return d[key].astype(np.float32), d["y"].astype(np.float32)


def build_head(in_dim: int, hidden: int, dropout: float):
    """임베딩 -> 스칼라 MLP(작게). sigmoid로 0~1 출력."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1), nn.Sigmoid(),
    )


def pairwise_rank_loss(pred, y, margin: float):
    """배치 내 모든 쌍(i,j) 중 y_i>y_j인 쌍에서 pred_i-pred_j가 margin 이상이도록 hinge."""
    import torch

    p = pred.view(-1)
    yy = y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)            # pred 차이 [N,N]
    dy = yy.unsqueeze(1) - yy.unsqueeze(0)          # 타깃 차이
    mask = (dy > 0).float()                         # y_i > y_j 인 쌍만
    denom = mask.sum().clamp(min=1.0)
    loss = (torch.relu(margin - dp) * mask).sum() / denom
    return loss


def evaluate(head, Xe, ye, device) -> dict:
    """eval 예측 -> Spearman + 분포 통계."""
    import torch

    head.eval()
    with torch.no_grad():
        pe = head(torch.from_numpy(Xe).to(device)).view(-1).cpu().numpy()
    rho = spearman(pe, ye)
    uniq = len({round(float(v), 3) for v in pe})
    return {"rho": rho, "pred": pe, "std": float(pe.std()),
            "min": float(pe.min()), "max": float(pe.max()), "uniq": uniq}


def main() -> None:
    ap = argparse.ArgumentParser(description="C-1 2단계: 회귀 헤드 + 랭킹 손실 판정")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio")
    ap.add_argument("--feat", default="concat", choices=["concat", "vis", "aud"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.1, help="랭킹 hinge 마진")
    ap.add_argument("--mse-w", type=float, default=0.3, help="MSE 가중(스케일 고정용)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.cache_dir)

    Xtr, ytr = load_split(out_dir, "train", args.feat)
    Xte, yte = load_split(out_dir, "eval", args.feat)
    # 표준화(train 통계로 train/eval 공통 적용)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    print(f"feat={args.feat} | train {Xtr.shape} eval {Xte.shape} | "
          f"ytr[std={ytr.std():.3f}] yte[std={yte.std():.3f}]")

    head = build_head(Xtr.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    n = Xtr.shape[0]

    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s: s + args.batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            pred = head(xb).view(-1)
            loss = pairwise_rank_loss(pred, yb, args.margin) + args.mse_w * mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 0 or ep == 1:
            ev = evaluate(head, Xte, yte, device)
            print(f"[ep {ep:3d}] loss={tot / n:.4f} | eval Spearman={ev['rho']:.4f} "
                  f"pred[min={ev['min']:.3f} max={ev['max']:.3f} std={ev['std']:.3f} "
                  f"uniq={ev['uniq']}]")

    ev = evaluate(head, Xte, yte, device)
    print("\n" + "=" * 60)
    print(f"=== C-1 최종 판정 (feat={args.feat}) ===")
    print(f"eval Spearman 상관: {ev['rho']:.4f}")
    print(f"예측 분포: min={ev['min']:.4f} max={ev['max']:.4f} std={ev['std']:.4f} "
          f"고유값={ev['uniq']}개")
    buckets = [0] * 10
    for v in ev["pred"]:
        buckets[min(int(v * 10), 9)] += 1
    for i, c in enumerate(buckets):
        print(f"  [{i / 10:.1f}~{(i + 1) / 10:.1f}) {c:4d} {'#' * int(40 * c / max(buckets))}")
    print("-" * 60)
    if ev["rho"] >= 0.3 and ev["std"] >= 0.1:
        print(f"해석: Spearman {ev['rho']:.2f} + 분포 퍼짐 -> 동결 임베딩에 신호 있음. "
              "0.111 붕괴는 LLM 출력 형식(거대 vocab CE) 탓. 전용 헤드 방향 정당화.")
    elif ev["rho"] < 0.15:
        print(f"해석: Spearman {ev['rho']:.2f} ~ 0 -> 동결 임베딩에 신호 거의 없음. "
              "아키텍처로 못 고침. transcript 입력 추가가 유일한 길.")
    else:
        print(f"해석: Spearman {ev['rho']:.2f} 약한 신호 -> 부분적. feat 바꿔(vis/aud) "
              "어느 모달이 기여하는지 확인 권장.")


if __name__ == "__main__":
    main()