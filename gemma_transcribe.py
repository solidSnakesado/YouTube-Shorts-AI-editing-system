# 46일차 신규 | 49일차 수정 2회째 | 배치: ~/project/yt_shorts_ai/gemma_transcribe.py
# 49일차 수정 이력 (수정본 기준 라인):
#   ①L48~63, L69~71, L74, L76~77 — 중단 재개 추가 (기존 출력 idx 건너뜀, append 모드)
#     사유: v2 전체 10,034클립 전사는 수 시간 소요 → 재개 필수
#   ②L118~120, L122~125 — 기본값을 v2 전체 전사로 변경
#     (train/eval_qwenfmt.jsonl, out-dir=gemma_audio_v2, limit 0=전체)
#   ③L26~31 — scripts/ 배치 대비 sys.path 레포 루트 추가
#
# 목적: B-1 1단계. 피벗 가설("raw audio가 transcript보다 나음 - 효과음/BGM까지 잡으니")을
#   검증하려면 transcript의 하이라이트 예측력을 raw audio(C-1 0.29)와 비교해야 함.
#   그 입력인 transcript를 Whisper로 뽑아 캐시한다(전사는 느려 한 번만 -> 2단계 재사용).
#
# 핵심: C-1 임베딩 npz와 결합하려면 같은 클립이 같은 순서여야 함 -> train/eval_numeric.jsonl
#   순서 그대로 따라가며 parse_sample로 audio_path 얻어 전사(C-1과 동일 경로/순서).
#   --limit로 샘플 규모 조절(train 1000 + eval 500 등). 결과: {idx,text,lang,prob,nwords} jsonl.
#
# 입력: train_numeric/eval_numeric.jsonl + datasets/gemma_audio/audio/*.wav
# 출력: transcript_train.jsonl / transcript_eval.jsonl (npz와 행 idx 정렬)
# 의존: faster_whisper, gemma_collate.parse_sample. 실행(GPU 권장):
#   python gemma_transcribe.py --limit-train 1000 --limit-eval 500
#   python gemma_transcribe.py   # 전체(느림)
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 49일차: 루트 배치 시
sys.path.insert(0, str(_HERE.parent))   # 49일차: scripts/ 배치 시 루트 추가
from gemma_collate import parse_sample  # noqa: E402


def load_rows(path: str, limit: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if 0 < limit <= len(rows):
                break
    return rows


def load_done_idxs(out_path: Path) -> set[int]:
    """49일차: 기존 출력에서 처리 완료 idx 집합 (중단 재개용)."""
    done: set[int] = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["idx"]))
            except Exception:
                continue
    return done


def transcribe_split(model, rows, base_dir, out_path: Path, tag: str) -> None:
    """각 행 audio 전사 -> {idx,text,lang,prob,nwords} 한 줄씩 기록(순서=npz idx)."""
    import os

    done = load_done_idxs(out_path)  # 49일차: 재개
    if done:
        print(f"  {tag}: 중단 재개 — 기존 {len(done)}건 건너뜀")
    empty = 0
    total_words = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if i in done:  # 49일차: 재개
                continue
            try:
                parsed = parse_sample(row)
                ap = parsed["audio_path"]
                if base_dir:
                    ap = os.path.join(base_dir, ap)
            except Exception as e:                  # noqa: BLE001
                rec = {"idx": i, "text": "", "lang": "", "prob": 0.0,
                       "nwords": 0, "err": f"parse:{type(e).__name__}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                empty += 1
                continue
            try:
                segs, info = model.transcribe(ap, beam_size=1, language=None)
                text = " ".join(s.text for s in segs).strip()
            except Exception as e:                  # noqa: BLE001
                text, info = "", None
                rec = {"idx": i, "text": "", "lang": "", "prob": 0.0,
                       "nwords": 0, "err": f"asr:{type(e).__name__}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                empty += 1
                continue
            nw = len(text.split())
            total_words += nw
            if nw == 0:
                empty += 1
            rec = {"idx": i, "text": text,
                   "lang": getattr(info, "language", ""),
                   "prob": round(float(getattr(info, "language_probability", 0.0)), 3),
                   "nwords": nw}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {tag}: {i + 1}/{len(rows)} (빈전사 {empty}, "
                      f"누적단어 {total_words})")
    print(f"  {tag} 완료: {len(rows)}건 | 빈전사 {empty} "
          f"({empty / max(len(rows), 1):.1%}) | 평균단어 {total_words / max(len(rows), 1):.1f}")
    print(f"  저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="B-1 1단계: 클립 Whisper 전사 캐시")
    ap.add_argument("--train", default="datasets/gemma_audio_v2/train_qwenfmt.jsonl")
    ap.add_argument("--eval", default="datasets/gemma_audio_v2/eval_qwenfmt.jsonl")
    ap.add_argument("--out-dir", default="datasets/gemma_audio_v2")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--model-size", default="small",
                    help="whisper 모델(small=0.15 프로브와 동일 조건, large-v3=정확)")
    ap.add_argument("--limit-train", type=int, default=0, help="0=전체")
    ap.add_argument("--limit-eval", type=int, default=0, help="0=전체")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    print(f"=== Whisper 로드: {args.model_size} ===")
    model = WhisperModel(args.model_size, device="cuda", compute_type="float16")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, path, lim in (("train", args.train, args.limit_train),
                             ("eval", args.eval, args.limit_eval)):
        rows = load_rows(path, lim)
        print(f"\n=== {split} 전사: {len(rows)}건 ({path}) ===")
        out_path = out_dir / f"transcript_{split}.jsonl"
        transcribe_split(model, rows, args.base_dir, out_path, split)

    print("\n완료. 다음: gemma_text_embed.py 로 transcript 임베딩 -> 헤드 비교.")


if __name__ == "__main__":
    main()