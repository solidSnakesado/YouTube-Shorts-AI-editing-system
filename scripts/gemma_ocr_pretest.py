# 49일차: gemma_ocr_pretest.py — 수정 2회
# 레포 경로: yt_shorts_ai/scripts/gemma_ocr_pretest.py
# 수정 이력 (수정본 기준 라인):
#   ①L25~27 — sys.path에 레포 루트 추가 (gemma_collate는 루트, gemma_e2e_collate는 scripts/)
#   ②L58, L70~80, L137~138, L156~157 — --upscale 옵션 추가
#     (해상도 병목 판별: 업스케일 후 인식률 개선 여부로 512px vs 원본화질 병목 구분)
# 역할: OCR 분기 진입 전 저비용 사전 검증 (로컬 WSL2, 12GB GPU)
#   (1) 인식률: 512px 프레임에서 EasyOCR(ko+en) 텍스트 검출 빈도·신뢰도
#   (2) 판별력: hook_score 상위/하위 그룹 간 OCR 텍스트 특성 차이
#       (박스 수·문자량·고신뢰 박스 수) — "읽히지만 판별력 없음" 조기 차단
#   (3) 비용: 프레임당 처리 시간 → 전 클립(10,526) 확장 비용 추정
# 실행 예 (레포 루트에서):
#   python scripts/gemma_ocr_pretest.py --n-clips 30 --frames-per-clip 3
# 출력: 콘솔 요약 + 텍스트 덤프 리포트(육안 대조용)

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 49일차: scripts/ (gemma_e2e_collate)
sys.path.insert(0, str(_HERE.parent))   # 49일차: 레포 루트 (gemma_collate)
from gemma_collate import parse_sample, sample_frames          # noqa: E402
from gemma_e2e_collate import extract_score, load_jsonl       # noqa: E402


# ---------------------------------------------------------------------------
# 1. 층화 샘플링 — 라벨 상위/하위 그룹
# ---------------------------------------------------------------------------
def stratified_pick(rows: list[dict], n_clips: int, seed: int):
    """hook_score 기준 상위 25% / 하위 25%에서 각 n_clips/2 무작위 추출."""
    scored = []
    for r in rows:
        try:
            p = parse_sample(r)
            scored.append((extract_score(p["target"]), p))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0])
    q = max(1, len(scored) // 4)
    low_pool, high_pool = scored[:q], scored[-q:]
    rng = random.Random(seed)
    k = n_clips // 2
    low = rng.sample(low_pool, min(k, len(low_pool)))
    high = rng.sample(high_pool, min(k, len(high_pool)))
    return high, low


# ---------------------------------------------------------------------------
# 2. 클립 단위 OCR
# ---------------------------------------------------------------------------
def ocr_clip(reader, parsed: dict, frames_per_clip: int, base_dir: Path,
             conf_thresh: float, upscale: float = 1.0):
    """클립 1개: 균등 샘플 프레임에 OCR → 프레임별 결과 리스트."""
    frames = sample_frames(parsed["frame_paths"], frames_per_clip)
    results = []
    for fp in frames:
        full = base_dir / fp
        if not full.exists():
            results.append({"frame": fp, "error": "파일 없음", "boxes": [],
                            "elapsed": 0.0})
            continue
        t0 = time.time()
        try:
            if upscale > 1.0:
                # 49일차(수정2회): 해상도 병목 판별용 — LANCZOS 업스케일 후 인식
                import numpy as np
                from PIL import Image
                img = Image.open(full).convert("RGB")
                w, h = img.size
                img = img.resize((int(w * upscale), int(h * upscale)),
                                 Image.LANCZOS)
                boxes = reader.readtext(np.array(img))
            else:
                boxes = reader.readtext(str(full))  # [(bbox, text, conf), ...]
        except Exception as e:
            results.append({"frame": fp, "error": str(e)[:80], "boxes": [],
                            "elapsed": 0.0})
            continue
        elapsed = time.time() - t0
        parsed_boxes = [{"text": t, "conf": float(c)} for _bb, t, c in boxes]
        results.append({
            "frame": fp,
            "error": None,
            "boxes": parsed_boxes,
            "n_boxes": len(parsed_boxes),
            "n_hi": sum(1 for b in parsed_boxes if b["conf"] >= conf_thresh),
            "n_chars": sum(len(b["text"]) for b in parsed_boxes),
            "elapsed": elapsed,
        })
    return results


# ---------------------------------------------------------------------------
# 3. 그룹 통계
# ---------------------------------------------------------------------------
def group_stats(clip_results: list[dict]) -> dict:
    """그룹(상위/하위) 프레임 단위 평균 지표."""
    frames = [f for c in clip_results for f in c["frames"] if f["error"] is None]
    if not frames:
        return {"frames": 0}
    return {
        "frames": len(frames),
        "boxes_mean": statistics.mean(f["n_boxes"] for f in frames),
        "hi_conf_mean": statistics.mean(f["n_hi"] for f in frames),
        "chars_mean": statistics.mean(f["n_chars"] for f in frames),
        "sec_mean": statistics.mean(f["elapsed"] for f in frames),
    }


def fmt_stats(name: str, s: dict) -> str:
    if s.get("frames", 0) == 0:
        return f"  [{name}] 유효 프레임 없음"
    return (f"  [{name}] 프레임 {s['frames']}장 | 박스/프레임 {s['boxes_mean']:.2f}"
            f" | 고신뢰 박스 {s['hi_conf_mean']:.2f} | 문자수 {s['chars_mean']:.1f}"
            f" | {s['sec_mean']:.2f}초/프레임")


# ---------------------------------------------------------------------------
# 4. 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="49일차: OCR 사전 테스트")
    ap.add_argument("--jsonl", default="datasets/gemma_audio_v2/train_qwenfmt.jsonl")
    ap.add_argument("--base-dir", default=".", help="jsonl 상대경로 기준 (레포 루트)")
    ap.add_argument("--n-clips", type=int, default=30, help="총 클립 수 (상/하위 반반)")
    ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--conf-thresh", type=float, default=0.4, help="고신뢰 박스 기준")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/ocr_pretest_report.txt")
    ap.add_argument("--cpu", action="store_true", help="GPU 미사용 (기본 GPU)")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="인식 전 업스케일 배율 (해상도 병목 판별용, 예: 2.0)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    rows = load_jsonl(str(base_dir / args.jsonl))
    print(f"jsonl {len(rows)}행 로드")

    high, low = stratified_pick(rows, args.n_clips, args.seed)
    print(f"층화 샘플: 상위 {len(high)}클립 / 하위 {len(low)}클립")

    import easyocr  # 49일차: 지연 import (정적 검증 통과용)
    print("EasyOCR 리더 초기화 (ko+en)...")
    reader = easyocr.Reader(["ko", "en"], gpu=not args.cpu)

    groups = {}
    for gname, picks in (("상위", high), ("하위", low)):
        clip_results = []
        for i, (score, parsed) in enumerate(picks, 1):
            frames = ocr_clip(reader, parsed, args.frames_per_clip,
                              base_dir, args.conf_thresh, args.upscale)
            clip_results.append({"score": score, "parsed": parsed,
                                 "frames": frames})
            done = sum(1 for f in frames if f["error"] is None)
            print(f"  [{gname} {i}/{len(picks)}] score={score:.2f}"
                  f" 프레임 {done}/{len(frames)} 처리")
        groups[gname] = clip_results

    # --- 요약 ---
    print("\n=== OCR 사전 테스트 요약 ===")
    all_stats = {}
    for gname, cr in groups.items():
        s = group_stats(cr)
        all_stats[gname] = s
        print(fmt_stats(gname, s))

    hi_s, lo_s = all_stats.get("상위", {}), all_stats.get("하위", {})
    if hi_s.get("frames") and lo_s.get("frames"):
        sec = statistics.mean([hi_s["sec_mean"], lo_s["sec_mean"]])
        total_h = sec * 10526 * 8 / 3600  # 전 클립 × max_frames 8
        print(f"\n  전 클립 확장 비용 추정: {sec:.2f}초/프레임"
              f" × 10,526클립 × 8프레임 ≈ {total_h:.1f}시간 (로컬 GPU)")
        print("  판별력 참고: 상위 vs 하위 박스/문자량 차이가 없으면"
              " OCR 신호가 라벨과 무관할 가능성 (덤프 육안 대조 필수)")

    # --- 육안 대조용 텍스트 덤프 ---
    out_path = base_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 49일차 OCR 사전 테스트 덤프 (육안 대조용)\n")
        f.write(f"# conf_thresh={args.conf_thresh}, seed={args.seed}\n\n")
        for gname, cr in groups.items():
            f.write(f"===== {gname} 그룹 =====\n")
            for c in cr:
                f.write(f"\n[score={c['score']:.2f}]"
                        f" audio={c['parsed']['audio_path']}\n")
                for fr in c["frames"]:
                    f.write(f"  frame: {fr['frame']}\n")
                    if fr["error"]:
                        f.write(f"    (오류: {fr['error']})\n")
                        continue
                    if not fr["boxes"]:
                        f.write("    (검출 없음)\n")
                    for b in fr["boxes"]:
                        mark = "★" if b["conf"] >= args.conf_thresh else " "
                        f.write(f"   {mark}[{b['conf']:.2f}] {b['text']}\n")
    print(f"\n덤프 저장: {out_path}")
    print("다음 단계: 덤프에서 게임 UI/자막/킬로그 판독 가능 여부 육안 확인")


if __name__ == "__main__":
    main()