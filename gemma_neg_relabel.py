# 44일차 수정 | 배치: ~/project/yt_shorts_ai/gemma_neg_relabel.py
# 수정2(pos 재라벨): L85 pos_miss/pos_scores · L99~115 pos 분기를 통과->heatmap window 평균 재라벨
#   (붕괴 원인=pos 100% 0.8~1.0 과밀집 std0.067) · L144~145 리포트 · L152~158 pos 분포 출력.
# 수정1(neg_max): L81~82 시그니처 · L84 neg_high · L110~112 neg>neg_max 제외 · L176 --neg-max.
# 추가: pos·neg 둘 다 heatmap window 평균으로 통합 회귀 라벨. neg만 0.5 초과 제외(pos는 제외 없음).
#
# 목적: []-붕괴 대응(옵션2) - neg 타겟을 빈 리스트 -> 실제 hook_score로 재라벨.
#   neg는 비피크 구간이라 그 구간 heatmap value 평균이 실제(낮은) 강도. pos가 피크
#   avg_value를 쓴 것과 동일 방식. 합성 노이즈/고정라벨(B-2) 둘 다 회피.
#
# 동작: neg 행의 metadata(video_id, clip_start/end) -> heatmaps_merged.jsonl의 그 영상
#   heatmap 곡선 조회 -> [clip_start,clip_end] 걸친 엔트리 value 평균 -> 새 타겟.
#   pos 행은 그대로 통과. 재다운로드/재추출 없음(타겟 텍스트만 교체).
#
# 실행 위치: 로컬 WSL(heatmap 데이터가 로컬). 원본 보존 -> 새 파일로 출력.
# 의존: 표준 라이브러리만.
# 실행:
#   python gemma_neg_relabel.py
#   python gemma_neg_relabel.py --keep-unmatched   # 매칭 실패 neg를 []로 유지(기본=드롭)
# 후속: 출력 jsonl 재번들 -> Colab 업로드 -> 재학습. (collapse_check는 라벨 임계 갱신 필요 - 다음 산출)
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_heatmaps(path: str) -> dict:
    """heatmaps jsonl -> {video_id: [{start_sec,end_sec,value}, ...]}. merged 중복은 마지막 우선."""
    hm: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec.get("video_id")
            curve = rec.get("heatmap")
            if vid and isinstance(curve, list) and curve:
                hm[vid] = curve
    return hm


def window_score(curve: list, cs: float, ce: float):
    """[cs,ce]에 걸친 heatmap 엔트리 value 평균(없으면 None). 균등폭이라 단순평균=pos와 동일 취지."""
    vals = [e["value"] for e in curve
            if e.get("end_sec", 0) > cs and e.get("start_sec", 0) < ce]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def classify(row: dict) -> str:
    """target -> 'pos' | 'neg' | 'bad'. neg=빈 highlights, pos=hook_score 有."""
    try:
        t = row["messages"][1]["content"][0]["text"]
        obj = json.loads(t)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "bad"
    hl = obj.get("highlights")
    if not isinstance(hl, list):
        return "bad"
    if len(hl) == 0:
        return "neg"
    if isinstance(hl[0], dict) and "hook_score" in hl[0]:
        return "pos"
    return "bad"


def histogram(scores: list[float]) -> None:
    """0.0~1.0 0.1단위 10구간 막대."""
    buckets = [0] * 10
    for s in scores:
        buckets[min(int(s * 10), 9)] += 1
    mx = max(buckets) or 1
    for i, c in enumerate(buckets):
        print(f"  [{i/10:.1f}~{(i+1)/10:.1f}) {c:5d} {'#' * int(40 * c / mx)}")


def relabel_file(in_path: str, out_path: str, hm: dict,
                 keep_unmatched: bool, neg_max: float) -> dict:
    """한 jsonl 재라벨 -> 새 파일. 통계 dict 반환. neg_max 초과 neg는 제외(라벨 노이즈)."""
    pos_n = neg_ok = neg_miss = neg_high = bad = 0
    pos_miss = 0
    neg_scores: list[float] = []
    pos_scores: list[float] = []
    samples: list[tuple] = []
    out_rows: list[dict] = []

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = classify(row)
            if kind == "pos":
                # 44일차: pos도 heatmap window 평균으로 재라벨(붕괴 원인=pos 100% 0.8~1.0
                #   과밀집 std0.067). neg와 동일 기준 -> 통합 회귀. pos는 상한 제외 없음.
                meta = row.get("metadata", {})
                vid, cs, ce = meta.get("video_id"), meta.get("clip_start"), meta.get("clip_end")
                curve = hm.get(vid) if vid else None
                score = (window_score(curve, cs, ce)
                         if curve and cs is not None and ce is not None else None)
                if score is None:
                    pos_miss += 1
                    out_rows.append(row)                # 매칭 실패 시 원본 라벨 유지
                    continue
                row["messages"][1]["content"][0]["text"] = json.dumps(
                    {"highlights": [{"hook_score": score}]}, ensure_ascii=False)
                pos_n += 1
                pos_scores.append(score)
                out_rows.append(row)
            elif kind == "neg":
                meta = row.get("metadata", {})
                vid, cs, ce = meta.get("video_id"), meta.get("clip_start"), meta.get("clip_end")
                curve = hm.get(vid) if vid else None
                score = (window_score(curve, cs, ce)
                         if curve and cs is not None and ce is not None else None)
                if score is None:
                    neg_miss += 1
                    if keep_unmatched:
                        out_rows.append(row)            # []로 유지
                    continue                            # 기본: 드롭
                if score > neg_max:                     # 44일차: neg인데 고값=비피크 오선택 -> 제외
                    neg_high += 1
                    continue
                row["messages"][1]["content"][0]["text"] = json.dumps(
                    {"highlights": [{"hook_score": score}]}, ensure_ascii=False)
                neg_ok += 1
                neg_scores.append(score)
                if len(samples) < 3:
                    samples.append((vid, cs, ce, score))
                out_rows.append(row)
            else:
                bad += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== {in_path} -> {out_path} ===")
    print(f"pos 재라벨 {pos_n}(매칭실패 {pos_miss}) | neg 재라벨 {neg_ok} | "
          f"neg>{neg_max} 제외 {neg_high} | neg 매칭실패 {neg_miss} | bad {bad}")
    bal = pos_n + neg_ok
    print(f"출력 행수: {len(out_rows)}  (매칭률 {neg_ok/(neg_ok+neg_miss):.1%}"
          f", pos/neg = {pos_n}:{neg_ok} = {pos_n/bal:.1%}:{neg_ok/bal:.1%})"
          if (neg_ok + neg_miss) and bal else f"출력 행수: {len(out_rows)}")
    if pos_scores:
        sd = statistics.pstdev(pos_scores) if len(pos_scores) > 1 else 0.0
        hi = sum(1 for s in pos_scores if s >= 0.8) / len(pos_scores)
        print(f"pos 점수: min={min(pos_scores):.4f} max={max(pos_scores):.4f} "
              f"mean={statistics.mean(pos_scores):.4f} std={sd:.4f} (0.8+: {hi:.1%})")
        histogram(pos_scores)
    if neg_scores:
        sd = statistics.pstdev(neg_scores) if len(neg_scores) > 1 else 0.0
        print(f"neg 점수: min={min(neg_scores):.4f} max={max(neg_scores):.4f} "
              f"mean={statistics.mean(neg_scores):.4f} std={sd:.4f}")
        histogram(neg_scores)
        print(" 샘플(vid, start, end, score):")
        for s in samples:
            print(f"   {s[0]}  {s[1]}~{s[2]}s -> {s[3]}")
    return {"pos": pos_n, "neg_ok": neg_ok, "neg_miss": neg_miss}


def main() -> None:
    ap = argparse.ArgumentParser(description="neg 타겟 실제 hook_score 재라벨(heatmap 구간 평균)")
    ap.add_argument("--heatmap", default="data/heatmaps/heatmaps_merged.jsonl")
    ap.add_argument("--in-train", default="datasets/gemma_audio/train.jsonl")
    ap.add_argument("--in-eval", default="datasets/gemma_audio/eval.jsonl")
    ap.add_argument("--out-train", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--out-eval", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="매칭 실패 neg를 []로 유지(기본=드롭)")
    ap.add_argument("--neg-max", type=float, default=0.5,
                    help="neg 점수 상한 초과 행 제외(비피크 오선택 라벨노이즈 제거). 기본 0.5")
    args = ap.parse_args()

    if not Path(args.heatmap).is_file():
        print(f"heatmap 없음: {args.heatmap}")
        return
    hm = load_heatmaps(args.heatmap)
    print(f"heatmap 로드: {len(hm)}개 영상")

    relabel_file(args.in_train, args.out_train, hm, args.keep_unmatched, args.neg_max)
    relabel_file(args.in_eval, args.out_eval, hm, args.keep_unmatched, args.neg_max)
    print("\n완료. 출력 파일 검토 후 재번들 -> Colab 업로드 -> 재학습.")


if __name__ == "__main__":
    main()