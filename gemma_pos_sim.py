# 44일차 수정 | 배치: ~/project/yt_shorts_ai/gemma_pos_sim.py
# 수정 라인(본 파일 기준): L27 import(classify 제거 - 미사용) ·
#   L48~51 pos 판별을 classify(점수기반,relabel neg 오분류) -> metadata.label 기반으로 교체
# 사유: classify가 점수 있으면 pos로 봐서 relabel neg(점수>0)를 pos로 오분류 -> 집계에 neg 혼입.
#
# 목적: round3 붕괴 원인=pos 라벨 100% 0.8~1.0 과밀집(std 0.067).
#   처방(A) "pos를 실제 heatmap 값으로 재라벨"이 정말 분산을 만드는지 측정만.
#   (파일 출력 X - 분포/통계만 찍어 (A) 채택 여부 판단)
#
#   현재 pos 라벨 = peak.avg_value(피크 세그먼트 평균) -> 0.8~1.0 몰림.
#   시뮬: pos 클립 [clip_start,clip_end]의 heatmap 곡선 평균(window_score)으로
#   대체했을 때 분포. 펼쳐지면 (A) 확정 / 여전히 몰리면 (B)/(C).
#
#   비교 출력:
#     - 현재 pos 라벨(target의 hook_score) 분포
#     - 시뮬 pos 라벨(heatmap window 평균) 분포
#     -> 둘을 나란히 봐서 분산 개선 여부 확인.
#
# 의존(번들): gemma_neg_relabel.py(load_heatmaps/window_score/classify/histogram),
#   gemma_collapse_check.py(parse_hook_score), gemma_collate.py(parse_sample).
# 실행:
#   python gemma_pos_sim.py --train datasets/gemma_audio/train_relabel.jsonl
from __future__ import annotations

import argparse
import json
import statistics

from gemma_collapse_check import parse_hook_score
from gemma_neg_relabel import histogram, load_heatmaps, window_score


def collect(train_path: str, hm: dict):
    """pos 행에서 (현재라벨, 시뮬라벨=heatmap window평균) 쌍 수집.

    현재라벨: target JSON의 hook_score(=peak.avg_value).
    시뮬라벨: metadata [clip_start,clip_end]의 heatmap 곡선 평균.
    매칭 실패(heatmap 없음/구간 밖)는 별도 카운트.
    """
    from gemma_collate import parse_sample

    cur: list[float] = []
    sim: list[float] = []
    miss = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # 44일차: classify는 점수있으면 pos로 봐서 relabel neg를 오분류 -> label 기반
            #   neg = metadata.label=="negative", 나머지(label 없음) = pos.
            if row.get("metadata", {}).get("label") == "negative":
                continue
            try:
                gt = parse_hook_score(parse_sample(row)["target"])
            except Exception:                       # noqa: BLE001
                continue
            if gt >= 0:
                cur.append(gt)
            meta = row.get("metadata", {})
            vid = meta.get("video_id")
            cs, ce = meta.get("clip_start"), meta.get("clip_end")
            curve = hm.get(vid) if vid else None
            sc = (window_score(curve, cs, ce)
                  if curve and cs is not None and ce is not None else None)
            if sc is None:
                miss += 1
            else:
                sim.append(sc)
    return cur, sim, miss


def summarize(name: str, scores: list[float]) -> None:
    """분포 통계 + 히스토그램 + 0.8~1.0 과밀집률."""
    print(f"\n=== {name} (n={len(scores)}) ===")
    if not scores:
        print("점수 없음")
        return
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"min={min(scores):.4f} max={max(scores):.4f} "
          f"mean={statistics.mean(scores):.4f} std={sd:.4f}")
    histogram(scores)
    hi = sum(1 for s in scores if s >= 0.8)
    print(f"  -> 0.8~1.0: {hi}/{len(scores)} = {hi/len(scores):.1%} "
          f"({'과밀집' if hi/len(scores) > 0.7 else '분산됨'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="pos 실제 heatmap 재라벨 분포 시뮬(측정만)")
    ap.add_argument("--train", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--heatmap", default="data/heatmaps/heatmaps_merged.jsonl")
    args = ap.parse_args()

    hm = load_heatmaps(args.heatmap)
    print(f"heatmap 로드: {len(hm)}개 영상")

    cur, sim, miss = collect(args.train, hm)
    summarize("현재 pos 라벨(peak.avg_value)", cur)
    summarize("시뮬 pos 라벨(heatmap window 평균)", sim)
    print(f"\nheatmap 매칭 실패 pos: {miss}")
    if sim:
        sd = statistics.pstdev(sim) if len(sim) > 1 else 0.0
        hi = sum(1 for s in sim if s >= 0.8) / len(sim)
        verdict = ("(A) 채택 가능 - pos 분산 개선" if sd > 0.12 or hi < 0.7
                   else "(A) 효과 약함 - (B)통합회귀/(C)loss 재검토")
        print(f"판정: {verdict}")


if __name__ == "__main__":
    main()