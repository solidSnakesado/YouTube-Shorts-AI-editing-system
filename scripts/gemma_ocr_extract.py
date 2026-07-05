# 49일차: gemma_ocr_extract.py — 신규 (전체 신규, 수정 0회)
# 레포 경로: yt_shorts_ai/scripts/gemma_ocr_extract.py
# 역할: OCR 프로브 게이트용 텍스트 추출 (로컬 WSL2 GPU)
#   - qwenfmt jsonl의 클립별 프레임(학습과 동일한 균등 샘플, 기본 8장)에
#     EasyOCR(ko+en) + 2.0x LANCZOS 업스케일(사전 테스트 확정 전처리) 적용
#   - 클립당 1행 jsonl 캐시 저장 (박스 전체 + conf 보존 — 필터는 프로브 단계에서)
#   - 중단 재개 지원: 기존 캐시의 clip_id는 건너뜀 (append)
# 실행 예 (레포 루트에서):
#   python scripts/gemma_ocr_extract.py \
#     --jsonl datasets/gemma_audio_v2/eval_qwenfmt.jsonl \
#     --out datasets/gemma_audio_v2/ocr_cache_eval.jsonl
# 다음 단계: gemma_ocr_probe.py (텍스트 임베딩 → Spearman 게이트)

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 49일차: scripts/ (gemma_e2e_collate)
sys.path.insert(0, str(_HERE.parent))   # 49일차: 레포 루트 (gemma_collate)
from gemma_collate import parse_sample, sample_frames          # noqa: E402
from gemma_e2e_collate import extract_score, load_jsonl       # noqa: E402


# ---------------------------------------------------------------------------
# 1. 유틸
# ---------------------------------------------------------------------------
def clip_id_of(parsed: dict) -> str:
    """오디오 경로 stem을 클립 ID로 사용 (예: 30Yh5bjCb24_13015)."""
    return Path(parsed["audio_path"]).stem


def load_done_ids(out_path: Path) -> set[str]:
    """기존 캐시에서 처리 완료된 clip_id 집합 (중단 재개용)."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                continue
    return done


# ---------------------------------------------------------------------------
# 2. 프레임 OCR (업스케일 2.0 표준)
# ---------------------------------------------------------------------------
def ocr_frame(reader, frame_path: Path, upscale: float):
    """프레임 1장 → 박스 리스트 [{text, conf}]. 실패 시 None."""
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(frame_path).convert("RGB")
        if upscale > 1.0:
            w, h = img.size
            img = img.resize((int(w * upscale), int(h * upscale)),
                             Image.LANCZOS)
        boxes = reader.readtext(np.array(img))
    except Exception:
        return None
    return [{"text": t, "conf": round(float(c), 3)} for _bb, t, c in boxes]


def process_clip(reader, parsed: dict, score: float, max_frames: int,
                 base_dir: Path, upscale: float) -> dict:
    """클립 1개 → 캐시 1행 dict."""
    frames = sample_frames(parsed["frame_paths"], max_frames)
    frame_records = []
    for fp in frames:
        full = base_dir / fp
        if not full.exists():
            frame_records.append({"frame": fp, "boxes": None})
            continue
        boxes = ocr_frame(reader, full, upscale)
        frame_records.append({"frame": fp, "boxes": boxes})
    return {
        "clip_id": clip_id_of(parsed),
        "score": score,
        "frames": frame_records,
    }


# ---------------------------------------------------------------------------
# 3. 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="49일차: OCR 텍스트 추출 (프로브 게이트용)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio_v2/eval_qwenfmt.jsonl")
    ap.add_argument("--out", default="datasets/gemma_audio_v2/ocr_cache_eval.jsonl")
    ap.add_argument("--base-dir", default=".", help="jsonl 상대경로 기준 (레포 루트)")
    ap.add_argument("--max-frames", type=int, default=8,
                    help="클립당 프레임 수 (학습과 동일 샘플링)")
    ap.add_argument("--upscale", type=float, default=2.0,
                    help="사전 테스트 확정 전처리 배율")
    ap.add_argument("--limit", type=int, default=0, help="0=전체, N=앞 N클립만")
    ap.add_argument("--cpu", action="store_true", help="GPU 미사용 (기본 GPU)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    rows = load_jsonl(str(base_dir / args.jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"jsonl {len(rows)}행 로드")

    out_path = base_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    if done:
        print(f"중단 재개: 기존 캐시 {len(done)}클립 건너뜀")

    import easyocr  # 49일차: 지연 import (정적 검증 통과용)
    print("EasyOCR 리더 초기화 (ko+en)...")
    reader = easyocr.Reader(["ko", "en"], gpu=not args.cpu)

    n_done, n_skip, n_err = 0, 0, 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            try:
                parsed = parse_sample(row)
                cid = clip_id_of(parsed)
                if cid in done:
                    n_skip += 1
                    continue
                score = extract_score(parsed["target"])
                rec = process_clip(reader, parsed, score, args.max_frames,
                                   base_dir, args.upscale)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"  [오류] {i}행: {str(e)[:80]}")
                continue
            if n_done % 100 == 0 and n_done > 0:
                el = time.time() - t0
                rate = n_done / el
                remain = (len(rows) - n_skip - n_done) / rate / 60
                print(f"  {n_done}클립 완료 ({rate:.1f}클립/초,"
                      f" 남은 예상 {remain:.0f}분)")
                f.flush()

    el = time.time() - t0
    print("\n=== 추출 완료 ===")
    print(f"  신규 {n_done} / 건너뜀 {n_skip} / 오류 {n_err}"
          f" | 소요 {el / 60:.1f}분")
    print(f"  캐시: {out_path}")


if __name__ == "__main__":
    main()