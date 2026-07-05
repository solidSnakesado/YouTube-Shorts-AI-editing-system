# 46일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_head_rank.py
# [수정 1회] --min-clips 옵션 추가(macro_spearman 필터 + main 인자). 사유: 클립 2개
#   영상은 Spearman이 ±1뿐이라 노이즈 큼 -> 클립 충분한 영상만으로 진짜 신호 강도 확인.
#
# 목적: 경로 A. 모든 진단이 0.3 천장 -> "영상마다 히트맵 기준이 달라(인기영상 전체↑,
#   비인기 전체↓) 전체를 한 줄로 세우면 영상간 스케일이 노이즈로 신호를 덮는가"를 시험.
#   학습/임베딩은 C-1 그대로, 평가만 영상 내 상대 랭킹(영상별 Spearman -> 평균)으로 바꾼다.
#
#   판정(전체 Spearman 0.26~0.30 대비):
#     - 영상내 macro Spearman >> 전체(예: 0.45+) -> 영상간 스케일이 범인. 태스크를
#       "영상 내 상대 랭킹"으로 정의하면 신호 살아있음. 실제 용도(한 영상서 베스트 구간
#       고르기)와도 정합 -> 이 방향으로 본격화.
#     - 전체와 비슷(~0.30) -> 영상간 노이즈도 아님. 입력 신호 한계 확정 -> B-1(transcript).
#
# 평가 방식: eval 클립을 video_id로 그룹화, 각 영상서 (예측, 타깃) Spearman 계산,
#   클립 2개+ 영상만, 영상별 값을 단순 평균(macro). 전체 Spearman(기존)도 같이 출력.
# 순서 검증: jsonl을 순서대로 읽어 y를 npz y와 대조(스킵0이라 일치 기대; 불일치시 경고).
#
# 입력: head_cache_train/eval.npz(C-1 평균풀링) + eval_numeric.jsonl(video_id). --feat 선택.
# 의존: numpy, torch, scipy(폴백 내장). 실행: python gemma_head_rank.py [--feat concat]
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load_split(out_dir: Path, split: str, feat: str):
    d = np.load(out_dir / f"head_cache_{split}.npz")
    key = {"concat": "X", "vis": "X_vis", "aud": "X_aud"}[feat]
    return d[key].astype(np.float32), d["y"].astype(np.float32)


def load_video_ids(jsonl: str, n_expect: int, y_npz: np.ndarray):
    """jsonl 순서대로 video_id + y 읽고, npz y와 대조(순서 일치 검증)."""
    vids, ys = [], []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vids.append(row.get("metadata", {}).get("video_id"))
            try:
                import re
                t = row["messages"][1]["content"][0]["text"]
                m = re.search(r"[01]?\.\d+|[01]", str(t))
                ys.append(float(m.group()) if m else None)
            except Exception:                       # noqa: BLE001
                ys.append(None)
    if len(vids) != n_expect:
        print(f"  ⚠️ jsonl 행수({len(vids)}) != npz N({n_expect}) -> 순서 매칭 불가")
        return None
    # y 대조(앞 50개)
    ok = sum(1 for i in range(min(50, n_expect))
             if ys[i] is not None and abs(ys[i] - float(y_npz[i])) < 1e-3)
    print(f"  순서 검증: 앞 50개 중 y 일치 {ok}/50 "
          f"({'정상' if ok >= 48 else '불일치 의심'})")
    return vids


def build_head(in_dim: int, hidden: int, dropout: float):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1), nn.Sigmoid(),
    )


def pairwise_rank_loss(pred, y, margin: float):
    import torch
    p, yy = pred.view(-1), y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = yy.unsqueeze(1) - yy.unsqueeze(0)
    mask = (dy > 0).float()
    return (torch.relu(margin - dp) * mask).sum() / mask.sum().clamp(min=1.0)


def macro_spearman(pred, y, vids, min_clips: int = 2):
    """영상별 Spearman -> 평균. 클립 min_clips+ 영상만. (영상수, 평균, 분포) 반환."""
    groups = defaultdict(list)
    for i, v in enumerate(vids):
        groups[v].append(i)
    rhos = []
    for v, idxs in groups.items():
        if len(idxs) < min_clips:
            continue
        pp = pred[idxs]
        yy = y[idxs]
        if np.std(yy) < 1e-6 or np.std(pp) < 1e-6:
            continue                                # 분산 0이면 상관 정의 안 됨
        rhos.append(spearman(pp, yy))
    return len(rhos), float(np.mean(rhos)) if rhos else 0.0, np.asarray(rhos)


def main() -> None:
    ap = argparse.ArgumentParser(description="A: 영상 내 상대 랭킹 평가")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio")
    ap.add_argument("--eval-jsonl", default="datasets/gemma_audio/eval_numeric.jsonl")
    ap.add_argument("--feat", default="concat", choices=["concat", "vis", "aud"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--mse-w", type=float, default=0.3)
    ap.add_argument("--min-clips", type=int, default=2,
                    help="이 클립수 미만 영상 제외(2=전체, 5=노이즈 적은 영상만)")
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
    vids = load_video_ids(args.eval_jsonl, len(yte), yte)
    if vids is None:
        print("순서 매칭 불가 -> 중단")
        return

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    print(f"feat={args.feat} | train {Xtr.shape} eval {Xte.shape} | 영상 {len(set(vids))}개")

    head = build_head(Xtr.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    n = Xtr.shape[0]

    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch):
            idx = perm[s: s + args.batch]
            pred = head(Xtr_t[idx]).view(-1)
            yb = ytr_t[idx]
            loss = pairwise_rank_loss(pred, yb, args.margin) + args.mse_w * mse(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        pe = head(torch.from_numpy(Xte).to(device)).view(-1).cpu().numpy()
    overall = spearman(pe, yte)
    nv, macro, rhos = macro_spearman(pe, yte, vids, args.min_clips)

    print("\n" + "=" * 60)
    print(f"=== A 판정: 영상 내 상대 랭킹 (feat={args.feat}, 클립{args.min_clips}+) ===")
    print(f"전체 Spearman (기존 방식)     : {overall:.4f}")
    print(f"영상 내 macro Spearman        : {macro:.4f}  (영상 {nv}개 평균)")
    if len(rhos):
        print(f"  영상별 분포: min={rhos.min():.3f} max={rhos.max():.3f} "
              f"중앙값={np.median(rhos):.3f} | >0.5인 영상 {int((rhos > 0.5).sum())}/{nv}")
    print("-" * 60)
    if macro >= overall + 0.12:
        print(f"해석: 영상내 {macro:.2f} >> 전체 {overall:.2f} -> 영상간 스케일이 노이즈였음. "
              "태스크를 영상 내 상대 랭킹으로 정의하면 신호 살아있음. 실용(한 영상서 베스트 "
              "구간 고르기)과 정합 -> 이 방향 본격화.")
    elif macro <= overall + 0.03:
        print(f"해석: 영상내 {macro:.2f} ~ 전체 {overall:.2f} -> 영상간 노이즈도 아님. "
              "입력 신호 한계 확정 -> 다음: B-1(transcript 시험).")
    else:
        print(f"해석: 영상내 {macro:.2f} 소폭 상승 -> 부분적. 영상별 분포 확인 후 판단.")


if __name__ == "__main__":
    main()