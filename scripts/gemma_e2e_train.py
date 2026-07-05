# 48일차: gemma_e2e_train.py — 신규 | 49일차 수정 1회째 | 51일차 수정 2회째
# 레포 경로: yt_shorts_ai/scripts/gemma_e2e_train.py
# 51일차 수정 이력 (수정본 기준 라인):
#   ①L129~138 — save_best 추가 (best 별도 보존, 49일차 step1100 0.298 유실 교훈)
#   ②L169~170 — --margin 기본값 0.1→1.0 (round12_m1 확정값, 실수 재발 방지)
#   ③L223, L254~L260 — 주기 평가에서 best 갱신 시 out/best/ 저장
#   ④L269~L283 — 최종 판정 블록을 2차 학습 기준으로 교체 (본 판정=OK-rate 명시)
# 49일차 수정 이력 (수정본 기준 라인):
#   ①L99~121 — 체크포인트 회전 추가 (rotate_ckpts + save_ckpt keep 인자)
#   ②L155~157 — --save-limit 인자 추가 (기본 2개 유지), L241·L244 호출부 전달
#   ③L247~259 — 최종 판정 블록을 round13 기준으로 교체
#     (round12 실측 0.2671 대비: +0.03 초과=주입 유효 / ±0.02 이내=주입 무효)
#   학습 로직 무수정 — round12와 동일 코드, 변인은 입력 jsonl(*_inj)만
# 역할: 방안 1(round12)/방안 2(round13) 학습 — Colab A100 전용
#   - 커스텀 학습 루프 (SFTTrainer 미사용: CE 전제라 부적합)
#   - 손실 = 랭킹(pairwise margin hinge, C-1/C-2 6회 무붕괴 검증 공식 그대로) + 약한 MSE
#   - 타깃 표준화(train mu/sd) → norm_stats.json으로 체크포인트에 동봉 (추론 정합)
#   - 주기 평가: eval Spearman + 예측 분포 (eval_loss 금지 — 핸드오프 §5 판정 기준)
#   - Drive 체크포인트 save_steps + --resume 재개 (런타임 끊김 대비)
#   - 랭킹 쌍은 실배치 내 비교 → batch>=8 권장 (accum과 별개)
# 실행(Colab):
#   python gemma_e2e_train.py \
#     --train /content/datasets/gemma_audio_v2/train_qwenfmt.jsonl \
#     --eval  /content/datasets/gemma_audio_v2/eval_qwenfmt.jsonl \
#     --out   /content/drive/MyDrive/gemma4_adapters/round12
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from gemma_e2e_collate import build_e2e_collate_fn, extract_score, load_jsonl
from gemma_e2e_model import count_trainable, load_model_for_train, save_checkpoint


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


def pairwise_rank_loss(pred, y, margin: float):
    """C-1/C-2 검증 공식 그대로: y_i>y_j 쌍에서 pred_i-pred_j >= margin hinge."""
    import torch

    p = pred.view(-1)
    yy = y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = yy.unsqueeze(1) - yy.unsqueeze(0)
    mask = (dy > 0).float()
    denom = mask.sum().clamp(min=1.0)
    return (torch.relu(margin - dp) * mask).sum() / denom


def to_device(batch: dict, device: str) -> dict:
    import torch
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# 평가: eval 서브셋 예측 → Spearman + 분포 (표준화 역변환 후 원 스케일로 보고)
# ---------------------------------------------------------------------------
def evaluate(model, rows: list[dict], collate, device: str,
             mu: float, sd: float, batch: int) -> dict:
    import torch

    model.eval()
    preds: list[float] = []
    ys: list[float] = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            b = to_device(collate(rows[s: s + batch]), device)
            p = model(b)                              # 표준화 스케일 예측
            preds.extend((p.float().cpu().numpy() * sd + mu).tolist())
            ys.extend(b["y"].cpu().numpy().tolist())
    model.train()
    pe, ye = np.array(preds), np.array(ys)
    mid = float(((pe >= 0.2) & (pe <= 0.7)).mean())
    return {"rho": spearman(pe, ye), "std": float(pe.std()),
            "min": float(pe.min()), "max": float(pe.max()),
            "uniq": len({round(float(v), 3) for v in pe}), "mid_ratio": mid}


def print_eval(tag: str, ev: dict) -> None:
    print(f"{tag} Spearman={ev['rho']:.4f} | pred[min={ev['min']:.3f} "
          f"max={ev['max']:.3f} std={ev['std']:.3f} uniq={ev['uniq']} "
          f"중간값비율={ev['mid_ratio']:.2f}]")


# ---------------------------------------------------------------------------
# 체크포인트: 어댑터+헤드+norm_stats(save_checkpoint) + 학습 상태(step/opt)
# ---------------------------------------------------------------------------
def rotate_ckpts(out_dir: str, keep: int) -> None:
    """49일차: checkpoint-N 오래된 것부터 삭제 (Drive 용량 보호). final은 제외."""
    import re as _re
    import shutil
    pat = _re.compile(r"^checkpoint-(\d+)$")
    cks = sorted(
        (int(m.group(1)), e) for e in os.listdir(out_dir)
        if (m := pat.match(e)) and os.path.isdir(os.path.join(out_dir, e)))
    for _step, name in cks[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(out_dir, name), ignore_errors=True)
        print(f"[ckpt] 회전 삭제: {name}")


def save_ckpt(model, opt, out_dir: str, step: int, norm: dict,
              keep: int = 0) -> None:
    import torch
    ck = os.path.join(out_dir, f"checkpoint-{step}")
    save_checkpoint(model, ck, norm)
    torch.save({"step": step, "optimizer": opt.state_dict()},
               os.path.join(ck, "train_state.pt"))
    print(f"[ckpt] step {step} → {ck}")
    if keep > 0:
        rotate_ckpts(out_dir, keep)  # 49일차: 회전


def save_best(model, out_dir: str, step: int, rho: float, norm: dict) -> None:
    """51일차: best 별도 보존 — 49일차 교훈(step1100 0.298 유실) 반영.
    주기 평가는 서브셋(±0.05 분산)이라 best/는 보험용, 우열 판정은 전체 eval로만."""
    import json as _json
    bd = os.path.join(out_dir, "best")
    save_checkpoint(model, bd, norm)
    with open(os.path.join(bd, "best_meta.json"), "w") as f:
        _json.dump({"step": step, "rho_subset": round(rho, 4)}, f)
    print(f"[best] step {step} rho(서브셋)={rho:.4f} → {bd}")


def load_resume(model, opt, resume_dir: str) -> int:
    """--resume: LoRA 가중치 + 헤드 + 옵티마이저 + step 복원."""
    import torch
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(os.path.join(resume_dir, "adapter_model.safetensors"))
    set_peft_model_state_dict(model.base, sd)
    head = torch.load(os.path.join(resume_dir, "regression_head.pt"),
                      map_location="cpu")
    model.head.load_state_dict(head)
    st = torch.load(os.path.join(resume_dir, "train_state.pt"),
                    map_location="cpu")
    opt.load_state_dict(st["optimizer"])
    print(f"[resume] {resume_dir} → step {st['step']}부터 재개")
    return int(st["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description="방안 1 round12: E2E 회귀 QLoRA 학습")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", dest="eval_path", required=True)
    ap.add_argument("--out", required=True, help="Drive 체크포인트 루트 (round12 격리)")
    ap.add_argument("--base-dir", default=None, help="jsonl 상대경로 기준 (/content)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="실배치(랭킹 쌍 필요, >=8 권장)")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="51일차: 기본 1.0 (round12_m1 확정값 — 0.1은 준상수 예측 허용)")
    ap.add_argument("--mse-w", type=float, default=0.3)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-limit", type=int, default=2,
                    help="49일차: 유지할 checkpoint-N 수 (0=무제한)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=200, help="주기 평가용 eval 서브셋 크기")
    ap.add_argument("--resume", default=None, help="checkpoint-N 디렉토리")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    # --- 데이터 ---
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval_path)
    rng = np.random.default_rng(args.seed)
    eval_sub = [eval_rows[i] for i in
                rng.choice(len(eval_rows), min(args.eval_n, len(eval_rows)),
                           replace=False)]
    # 48일차: 타깃 표준화 통계 (train 전체에서 1회 산출 → 체크포인트 동봉)
    ytr = np.array([extract_score(r["messages"][1]["content"][0]["text"])
                    for r in train_rows], dtype=np.float32)
    norm = {"mu": float(ytr.mean()), "sd": float(ytr.std() + 1e-6)}
    print(f"train {len(train_rows)} / eval {len(eval_rows)} (서브셋 {len(eval_sub)}) | "
          f"y mu={norm['mu']:.3f} sd={norm['sd']:.3f}")

    # --- 모델 ---
    model, processor = load_model_for_train()
    model.head.to(device)
    print(count_trainable(model))
    collate = build_e2e_collate_fn(processor, max_frames=args.max_frames,
                                   base_dir=args.base_dir)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.base.parameters() if p.requires_grad],
         "lr": args.lr_lora},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    start_step = load_resume(model, opt, args.resume) if args.resume else 0

    # --- 학습 루프 ---
    mse = torch.nn.MSELoss()
    n = len(train_rows)
    steps_per_ep = (n + args.batch - 1) // args.batch
    total_steps = steps_per_ep * args.epochs
    step = start_step
    best_rho = float("-inf")        # 51일차: best 추적 (재개 시 초기화 — 재저장 무해)
    t0 = time.time()
    model.train()
    print(f"총 {total_steps} step ({steps_per_ep}/ep × {args.epochs}ep), "
          f"batch={args.batch}, margin={args.margin}, mse_w={args.mse_w}")

    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        for s in range(0, n, args.batch):
            # 재개 시 이미 지난 step 건너뜀 (데이터 순서는 seed 고정으로 재현)
            if (s // args.batch) + (ep - 1) * steps_per_ep < start_step:
                continue
            rows = [train_rows[i] for i in perm[s: s + args.batch]]
            if len(rows) < 2:
                continue                             # 랭킹 쌍 불가 배치 스킵
            batch = to_device(collate(rows), device)
            yz = (batch["y"] - norm["mu"]) / norm["sd"]   # 표준화 타깃
            pred = model(batch)
            loss = (pairwise_rank_loss(pred, yz, args.margin)
                    + args.mse_w * mse(pred, yz))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1

            if step % 20 == 0:
                el = time.time() - t0
                print(f"[ep{ep} step {step}/{total_steps}] loss={float(loss):.4f} "
                      f"({el / 60:.1f}분 경과)")
            if step % args.eval_steps == 0:
                ev = evaluate(model, eval_sub, collate, device,
                              norm["mu"], norm["sd"], args.batch)
                print_eval(f"  [eval@{step}]", ev)
                if ev["rho"] > best_rho:        # 51일차: best 별도 보존
                    best_rho = ev["rho"]
                    save_best(model, args.out, step, best_rho, norm)
            if step % args.save_steps == 0:
                save_ckpt(model, opt, args.out, step, norm, args.save_limit)

    # --- 최종: 전체 eval + final 저장 ---
    save_ckpt(model, opt, args.out, step, norm, args.save_limit)
    save_checkpoint(model, os.path.join(args.out, "final"), norm)
    ev = evaluate(model, eval_rows, collate, device,
                  norm["mu"], norm["sd"], args.batch)
    print("\n" + "=" * 60)
    print("=== 2차 학습 최종 판정 (전체 eval, 대조군 round12_m1=0.2671) ===")
    print_eval("final", ev)
    # 51일차: 사전 고정 판정 기준 — Spearman은 보조 지표, 본 판정은 OK-rate 재측정
    #   (피드백 202행은 train의 2.5%라 eval Spearman 무변도 정상 — hard sample의
    #    효과는 중간 점수대 변별력이므로 실전 OK율에서 드러남)
    if ev["rho"] > 0.2671 + 0.03:
        print("해석: 유의 개선 — 어댑터 채택 후보. OK-rate 재측정으로 확정.")
    elif ev["rho"] >= 0.2671 - 0.02:
        print("해석: 동률 이상 — 회귀(악화) 없음. OK-rate 재측정으로 본 판정 진행.")
    else:
        print("해석: 악화 — 분포(std/uniq/중간값비율) 확인. 업샘플 과다(평균 회귀) 의심 시 x1 재시도.")
    if best_rho > float("-inf"):
        print(f"[best] 서브셋 최고 rho={best_rho:.4f} → {os.path.join(args.out, 'best')} "
              f"(final과 별개 보존 — 필요 시 전체 eval로 재검증)")


if __name__ == "__main__":
    main()