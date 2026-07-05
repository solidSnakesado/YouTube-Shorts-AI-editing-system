# 48일차: gemma_e2e_infer.py — 수정 2회
#   수정 1회(48일차): L100 포맷 스펙 전각 문자 수정
#   수정 2회(50일차): --max-seq 인자 추가 + load_model_for_infer에 전달
#     (로컬 12GB에서 max_seq_length 8192 기본값이 오프로딩 유발 의심 → 3072로 실험, 기각됨)
#   수정 3회(50일차): 오프로딩 원인 분해용 계측 추가 (로드 후 VRAM + 샘플별 시간)
#   수정 4회(50일차): 샘플별 시간을 collate(CPU)/forward(GPU)로 분해 (수정본 기준 L83~99)
#     ① 로드 직후 VRAM(allocated/reserved) 출력  ② 매 샘플 진행+소요시간 출력(기존 10배치 간격)
# 레포 경로: yt_shorts_ai/scripts/gemma_e2e_infer.py
# 역할: 방안 1(round12) 추론 + 분포/Spearman 판정 — 독립 실행 스크립트
#   - 로컬 12GB VRAM 대응: 4bit base + LoRA 어댑터 + 회귀 헤드 (batch 기본 1)
#   - VRAM 잔류 대응: 독립 프로세스로 실행 → 종료 시 VRAM 반환
#     (실행 후 nvidia-smi로 0 MiB 확인, 잔류 시 kill -9 <PID>)
#   - norm_stats.json(mu/sd)로 표준화 역변환 → 원 스케일 hook_score 출력
#   - 판정 출력: Spearman + 분포 히스토그램 + 중간값 비율 (eval_loss 미사용)
# 실행(로컬 또는 Colab):
#   python gemma_e2e_infer.py \
#     --adapter /path/to/round12/final \
#     --eval datasets/gemma_audio_v2/eval_qwenfmt.jsonl --n 100
from __future__ import annotations

import argparse

import numpy as np

from gemma_e2e_collate import build_e2e_collate_fn, load_jsonl
from gemma_e2e_model import load_model_for_infer, load_norm_stats


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관 (gemma_head_train.py와 동일 — scipy 폴백 포함)."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="방안 1 round12: 로컬 추론 + 분포 판정")
    ap.add_argument("--adapter", required=True, help="checkpoint-N 또는 final 경로")
    ap.add_argument("--eval", dest="eval_path", required=True)
    ap.add_argument("--base-dir", default=None, help="jsonl 상대경로 기준 디렉토리")
    ap.add_argument("--n", type=int, default=100, help="평가 샘플 수 (0=전체)")
    ap.add_argument("--batch", type=int, default=1, help="로컬 12GB는 1 권장")
    ap.add_argument("--max-frames", type=int, default=8, help="학습 시 값과 동일해야 함")
    ap.add_argument("--max-seq", type=int, default=3072,
                    help="50일차: 로컬 12GB 오프로딩 회피용 (실입력 ~3,055토큰)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    # --- 데이터 (seed 고정 서브샘플 → 체크포인트 간 비교 재현성) ---
    rows = load_jsonl(args.eval_path)
    if args.n and args.n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), args.n, replace=False)]
    print(f"eval 샘플 {len(rows)}개 | adapter={args.adapter}")

    # --- 모델 + 정규화 통계 ---
    model, processor = load_model_for_infer(args.adapter, max_seq_length=args.max_seq)
    model.head.to(device)
    # 50일차: 로드 직후 가중치 발자국 실측 (오프로딩 원인 분해용)
    if device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        print(f"로드 후 VRAM: allocated={alloc:.2f}GB reserved={resv:.2f}GB")
    norm = load_norm_stats(args.adapter)
    print(f"norm_stats: mu={norm['mu']:.3f} sd={norm['sd']:.3f}")
    collate = build_e2e_collate_fn(processor, max_frames=args.max_frames,
                                   base_dir=args.base_dir)

    # --- 추론 (표준화 역변환 → 원 스케일) ---
    preds: list[float] = []
    ys: list[float] = []
    import time                                     # 50일차: 샘플별 시간 실측
    with torch.no_grad():
        for s in range(0, len(rows), args.batch):
            # 50일차 수정 4회: 샘플당 시간을 collate(CPU 전처리) / forward(GPU)로 분해
            t0 = time.time()
            batch = collate(rows[s: s + args.batch])
            t1 = time.time()
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            p = model(batch).float().cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()            # forward 시간 정확 측정
            t2 = time.time()
            preds.extend((p * norm["sd"] + norm["mu"]).tolist())
            ys.extend(batch["y"].cpu().numpy().tolist())
            print(f"  진행 {s + len(batch['y'])}/{len(rows)} "
                  f"(collate {t1 - t0:.1f}s + forward {t2 - t1:.1f}s)", flush=True)

    pe, ye = np.array(preds), np.array(ys)

    # --- 판정 출력 (round11 강화판 분포체크와 동일 지표) ---
    rho = spearman(pe, ye)
    uniq = len({round(float(v), 3) for v in pe})
    mid = float(((pe >= 0.2) & (pe <= 0.7)).mean())
    print("\n" + "=" * 60)
    print("=== round12 추론 분포 판정 ===")
    print(f"Spearman: {rho:.4f}")
    print(f"예측 분포: min={pe.min():.3f} max={pe.max():.3f} std={pe.std():.3f} "
          f"고유값={uniq}개 중간값비율={mid:.2f}")
    lo, hi = min(0.0, float(pe.min())), max(1.0, float(pe.max()))
    buckets = [0] * 10
    for v in pe:
        buckets[min(int((v - lo) / (hi - lo + 1e-9) * 10), 9)] += 1
    for i, c in enumerate(buckets):
        b0, b1 = lo + (hi - lo) * i / 10, lo + (hi - lo) * (i + 1) / 10
        print(f"  [{b0:5.2f}~{b1:5.2f}) {c:4d} {'#' * int(40 * c / max(buckets))}")
    print("-" * 60)
    if rho >= 0.35 and pe.std() >= 0.1:
        print("해석: 동결 천장 0.236 유의 상회 + 분포 퍼짐 → 층3 공략 성공.")
    elif rho <= 0.27:
        print("해석: 동결 천장 수준 정체 → OCR 분기 검토.")
    else:
        print("해석: 경계 구간 — 오분류 세부(고타깃→저예측) 확인 권장.")


if __name__ == "__main__":
    main()