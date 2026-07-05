# 49일차: gemma_ocr_probe.py — 신규 (전체 신규, 수정 0회)
# 레포 경로: yt_shorts_ai/scripts/gemma_ocr_probe.py
# 역할: OCR 분기 게이트 판정 (로컬)
#   - OCR 캐시(jsonl) → 클립별 텍스트 결합(conf 필터 + 중복 제거)
#   - 46일차 전사 프로브(gemma_text_embed.py)와 동일 방법론으로 임베딩·헤드 학습
#     (함수 직접 import: spearman / load_base / embed_texts / train_eval)
#   - 게이트: OCR 단독 Spearman ≥ 0.15(전사 기준) → 지시문 주입 재학습 진행
#             미만 → OCR 분기 종료(방안 2 검토)
#   - C-1 head_cache가 있으면 audio+OCR / vis+aud+OCR 결합 이득도 측정
#   - 임베딩은 npz 캐시 (재실행 시 임베딩 생략)
# 실행 예 (레포 루트에서):
#   python scripts/gemma_ocr_probe.py
# GPU: 로컬 12GB — device_map=auto로 부족분 CPU 오프로드(느려질 수 있음)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 49일차: scripts/
sys.path.insert(0, str(_HERE.parent))   # 49일차: 레포 루트 (gemma_text_embed)
from gemma_text_embed import embed_texts, load_base, train_eval  # noqa: E402


# ---------------------------------------------------------------------------
# 1. OCR 캐시 → 클립 텍스트
# ---------------------------------------------------------------------------
def build_text(rec: dict, min_conf: float) -> str:
    """캐시 1행 → 결합 텍스트.
    conf ≥ min_conf 박스만 사용, 클립 내 완전 중복 문자열 제거(순서 보존).
    중복 제거 근거: 상수 HUD 텍스트가 8프레임 반복되며 256토큰 예산을 잠식하고
    임베딩을 상수 신호로 지배하는 것을 방지."""
    seen: set[str] = set()
    parts: list[str] = []
    for fr in rec.get("frames") or []:
        for b in fr.get("boxes") or []:
            t = (b.get("text") or "").strip()
            if not t or b.get("conf", 0.0) < min_conf:
                continue
            if t in seen:
                continue
            seen.add(t)
            parts.append(t)
    return " ".join(parts)


def load_ocr_cache(path: Path, min_conf: float):
    """캐시 jsonl → (clip_ids, texts, y)."""
    ids, texts, ys = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec["clip_id"])
            texts.append(build_text(rec, min_conf))
            ys.append(float(rec["score"]))
    return ids, texts, np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. 임베딩 (npz 캐시)
# ---------------------------------------------------------------------------
def get_embeddings(texts_tr, texts_te, cache_path: Path, base_model: str):
    """임베딩 npz 캐시가 있으면 로드, 없으면 생성 후 저장."""
    if cache_path.exists():
        d = np.load(cache_path)
        if d["X_tr"].shape[0] == len(texts_tr) and \
           d["X_te"].shape[0] == len(texts_te):
            print(f"임베딩 캐시 로드: {cache_path}")
            return d["X_tr"], d["X_te"]
        print("임베딩 캐시 크기 불일치 → 재생성")
    print("=== base 로드 + OCR 텍스트 임베딩 (시간 소요) ===")
    model, processor = load_base(base_model)
    X_tr = embed_texts(model, processor, texts_tr)
    X_te = embed_texts(model, processor, texts_te)
    np.savez(cache_path, X_tr=X_tr, X_te=X_te)
    print(f"임베딩 캐시 저장: {cache_path}")
    return X_tr, X_te


# ---------------------------------------------------------------------------
# 3. C-1 결합 (정렬 검증 포함)
# ---------------------------------------------------------------------------
def try_load_head_cache(cd: Path, y_tr, y_te):
    """head_cache npz 로드 + y 정렬 검증. 실패 시 None (결합 프로브 생략)."""
    ptr, pte = cd / "head_cache_train.npz", cd / "head_cache_eval.npz"
    if not (ptr.exists() and pte.exists()):
        print("head_cache 없음 → 결합 프로브 생략 (OCR 단독만)")
        return None
    dtr, dte = np.load(ptr), np.load(pte)
    if dtr["y"].shape[0] != y_tr.shape[0] or dte["y"].shape[0] != y_te.shape[0]:
        print("head_cache 표본 수 불일치 → 결합 프로브 생략")
        return None
    if not (np.allclose(dtr["y"].astype(np.float32), y_tr, atol=1e-3)
            and np.allclose(dte["y"].astype(np.float32), y_te, atol=1e-3)):
        print("head_cache y 정렬 불일치 → 결합 프로브 생략 (순서 다름 의심)")
        return None
    return dtr, dte


# ---------------------------------------------------------------------------
# 4. 메인
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="49일차: OCR Spearman 프로브 게이트")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio_v2")
    ap.add_argument("--cache-train", default="ocr_cache_train.jsonl")
    ap.add_argument("--cache-eval", default="ocr_cache_eval.jsonl")
    ap.add_argument("--base-model", default="unsloth/gemma-4-E4B-it")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="박스 채택 conf 하한 (사전 테스트: 0.3 미만은 잡음 다수)")
    ap.add_argument("--gate", type=float, default=0.15,
                    help="게이트 기준 (46일차 전사 프로브 실측)")
    args = ap.parse_args()

    cd = Path(args.cache_dir)
    ids_tr, texts_tr, y_tr = load_ocr_cache(cd / args.cache_train, args.min_conf)
    ids_te, texts_te, y_te = load_ocr_cache(cd / args.cache_eval, args.min_conf)
    n_empty_tr = sum(1 for t in texts_tr if not t)
    n_empty_te = sum(1 for t in texts_te if not t)
    print(f"샘플: train {len(ids_tr)} (빈 텍스트 {n_empty_tr},"
          f" {100 * n_empty_tr / max(1, len(ids_tr)):.0f}%)"
          f" / eval {len(ids_te)} (빈 텍스트 {n_empty_te},"
          f" {100 * n_empty_te / max(1, len(ids_te)):.0f}%)")

    emb_cache = cd / f"ocr_embed_conf{args.min_conf:.2f}.npz"
    X_tr, X_te = get_embeddings(texts_tr, texts_te, emb_cache, args.base_model)

    print("\n" + "=" * 60)
    print(f"=== OCR 프로브 게이트 (n_eval={len(ids_te)}, 기준 {args.gate}) ===")
    r_ocr = train_eval(X_tr, y_tr, X_te, y_te, "OCR only")

    hc = try_load_head_cache(cd, y_tr, y_te)
    if hc is not None:
        dtr, dte = hc
        Xa_tr, Xa_te = dtr["X_aud"], dte["X_aud"]
        Xv_tr, Xv_te = dtr["X_vis"], dte["X_vis"]
        r_aud = train_eval(Xa_tr, y_tr, Xa_te, y_te, "audio only (대조)")
        r_ao = train_eval(np.concatenate([Xa_tr, X_tr], 1), y_tr,
                          np.concatenate([Xa_te, X_te], 1), y_te, "audio+OCR")
        r_all = train_eval(
            np.concatenate([Xv_tr, Xa_tr, X_tr], 1), y_tr,
            np.concatenate([Xv_te, Xa_te, X_te], 1), y_te, "vis+aud+OCR")
        print("-" * 60)
        if r_ao > r_aud + 0.03 or r_all > r_aud + 0.03:
            print(f"결합 이득 있음: audio {r_aud:.3f} → +OCR {r_ao:.3f},"
                  f" all {r_all:.3f} (직교 신호 존재)")
        else:
            print(f"결합 이득 미미: audio {r_aud:.3f} vs +OCR {r_ao:.3f},"
                  f" all {r_all:.3f} (오디오와 중복 의심)")

    print("-" * 60)
    if r_ocr >= args.gate:
        print(f"게이트 통과: OCR 단독 {r_ocr:.3f} ≥ {args.gate}"
              " → 지시문 주입 재학습 진행")
    else:
        print(f"게이트 미달: OCR 단독 {r_ocr:.3f} < {args.gate}"
              " → OCR 분기 종료, 방안 2 검토")


if __name__ == "__main__":
    main()