# 49일차: gemma_inj_builder.py — 신규 (전체 신규, 수정 0회)
# 레포 경로: yt_shorts_ai/scripts/gemma_inj_builder.py
# 역할: round13(방안 2) 주입 데이터셋 생성 (로컬, GPU 불필요)
#   - train/eval_qwenfmt.jsonl의 instruction에 전사(Whisper small)와
#     OCR 텍스트(conf≥0.3, 클립 내 중복 제거)를 덧붙여 *_inj.jsonl 생성
#   - collate/모델/학습 코드는 무수정 — round12와의 변인을 입력 jsonl로 국한
#   - 정렬 검증: 전사는 idx=행 순서, OCR은 clip_id=오디오 stem 대조 (불일치 시 중단)
#   - 텍스트 상한: 전사 80단어 / OCR 60단어 (max_length 3072→3584 상향 전제)
# 실행 예 (레포 루트에서):
#   python scripts/gemma_inj_builder.py
# 출력: datasets/gemma_audio_v2/train_inj.jsonl / eval_inj.jsonl + 커버리지 통계

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 49일차: scripts/
sys.path.insert(0, str(_HERE.parent))   # 49일차: 레포 루트 (gemma_collate)
from gemma_collate import parse_sample  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 캐시 로더
# ---------------------------------------------------------------------------
def load_jsonl_lines(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_transcripts(path: Path, n_expect: int) -> list[str]:
    """transcript jsonl → idx 순 텍스트 리스트 (개수 검증)."""
    recs = load_jsonl_lines(path)
    if len(recs) != n_expect:
        raise SystemExit(f"[중단] 전사 개수 불일치: {len(recs)} != {n_expect}")
    out = [""] * n_expect
    for r in recs:
        out[int(r["idx"])] = (r.get("text") or "").strip()
    return out


def load_ocr_texts(path: Path, min_conf: float) -> dict[str, str]:
    """OCR 캐시 → {clip_id: 결합 텍스트} (conf 필터 + 클립 내 중복 제거)."""
    out: dict[str, str] = {}
    for rec in load_jsonl_lines(path):
        seen: set[str] = set()
        parts: list[str] = []
        for fr in rec.get("frames") or []:
            for b in fr.get("boxes") or []:
                t = (b.get("text") or "").strip()
                if not t or b.get("conf", 0.0) < min_conf or t in seen:
                    continue
                seen.add(t)
                parts.append(t)
        out[rec["clip_id"]] = " ".join(parts)
    return out


def cap_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# 2. 주입 블록 구성
# ---------------------------------------------------------------------------
def build_inj_block(transcript: str, ocr: str,
                    max_tr: int, max_ocr: int) -> str:
    """instruction 뒤에 덧붙일 주입 블록. 없으면 '(없음)' 명시
    (모델이 '텍스트 부재'도 신호로 쓰도록 필드 자체는 항상 포함)."""
    tr = cap_words(transcript, max_tr) or "(없음)"
    oc = cap_words(ocr, max_ocr) or "(없음)"
    return (f"\n\n[음성 전사]\n{tr}"
            f"\n[화면 텍스트 OCR]\n{oc}")


def inject_split(qwen_path: Path, tr_path: Path, ocr_path: Path,
                 out_path: Path, min_conf: float,
                 max_tr: int, max_ocr: int, tag: str) -> None:
    rows = load_jsonl_lines(qwen_path)
    transcripts = load_transcripts(tr_path, len(rows))
    ocr_map = load_ocr_texts(ocr_path, min_conf)

    n_tr, n_ocr, n_both, n_none, n_mismatch = 0, 0, 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            parsed = parse_sample(row)
            cid = Path(parsed["audio_path"]).stem
            if cid not in ocr_map:
                n_mismatch += 1
                if n_mismatch <= 3:
                    print(f"  [경고] OCR 캐시에 없는 clip_id: {cid}")
            tr, oc = transcripts[i], ocr_map.get(cid, "")
            n_tr += bool(tr)
            n_ocr += bool(oc)
            n_both += bool(tr) and bool(oc)
            n_none += (not tr) and (not oc)

            # 49일차: qwenfmt 원행의 user 텍스트 블록(=instruction)에 블록 덧붙임
            new_row = json.loads(json.dumps(row, ensure_ascii=False))  # deep copy
            replaced = False
            for msg in new_row.get("messages", []):
                if msg.get("role") != "user":
                    continue
                for blk in msg.get("content", []):
                    if blk.get("type") == "text":
                        blk["text"] = blk["text"] + build_inj_block(
                            tr, oc, max_tr, max_ocr)
                        replaced = True
            if not replaced:
                raise SystemExit(f"[중단] {tag} {i}행: user 텍스트 블록 없음")
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"  {tag} 완료: {n}행 → {out_path.name}")
    print(f"    전사 있음 {n_tr} ({100 * n_tr / n:.0f}%)"
          f" | OCR 있음 {n_ocr} ({100 * n_ocr / n:.0f}%)"
          f" | 둘 다 {n_both} ({100 * n_both / n:.0f}%)"
          f" | 둘 다 없음 {n_none} ({100 * n_none / n:.0f}%)")
    if n_mismatch:
        print(f"    [경고] OCR clip_id 불일치 {n_mismatch}건 — 0이어야 정상")


# ---------------------------------------------------------------------------
# 3. 메인
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="49일차: round13 주입 jsonl 생성")
    ap.add_argument("--cache-dir", default="datasets/gemma_audio_v2")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--max-tr-words", type=int, default=80)
    ap.add_argument("--max-ocr-words", type=int, default=60)
    args = ap.parse_args()

    cd = Path(args.cache_dir)
    for split in ("train", "eval"):
        inject_split(
            qwen_path=cd / f"{split}_qwenfmt.jsonl",
            tr_path=cd / f"transcript_{split}.jsonl",
            ocr_path=cd / f"ocr_cache_{split}.jsonl",
            out_path=cd / f"{split}_inj.jsonl",
            min_conf=args.min_conf,
            max_tr=args.max_tr_words,
            max_ocr=args.max_ocr_words,
            tag=split,
        )
    print("\n완료. 다음: 첫 행 육안 검증 후 Drive 업로드 → round13 학습")


if __name__ == "__main__":
    main()