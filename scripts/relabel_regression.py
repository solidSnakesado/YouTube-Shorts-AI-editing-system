# 계층: 스크립트 (CLI 진입점)
# 역할: 기존 combined 데이터셋의 output을 실제 히트맵 참여도 점수로 재라벨링 (신호 검증용)
# 의존: heatmaps_merged.jsonl, dataset_generator_p2_combined.jsonl
# 32일차 신규: 이진 분류({"highlights":[]}/JSON) -> Verbalized 회귀({"engagement_score":X})
#   기존 프레임 재활용 (재다운로드 불필요), metadata 보존하여 라이브 train_data_loader 호환

"""회귀 재라벨링 - 각 클립의 output을 [clip_start, clip_end] 구간의 실제 히트맵 평균값으로 교체"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

# 32일차: 회귀 태스크 지시문 (하이라이트 추출 -> 참여도 점수 예측)
REGRESSION_INSTRUCTION = (
    "영상 프레임과 전사 텍스트를 분석하여 이 구간이 시청자에게 얼마나 다시 보고 싶은 "
    "구간인지 0.00~1.00 사이 참여도 점수로 예측하세요. "
    "{\"engagement_score\": 점수} 형식의 JSON으로만 반환하세요."
)

def load_heatmaps(path: Path) -> dict:
    """히트맵 JSONL -> {video_id: [세그먼트]}"""

    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                lookup[rec["video_id"]] = rec.get("heatmap", [])
            except (json.JSONDecodeError, KeyError):
                continue
    return lookup

def clip_engagement(segments: list, clip_start: float, clip_end: float) -> float | None:
    """클립 구간 겹치는 히트맵 세그먼트의 중첩 길이 가중 평균"""
    
    total_overlap = 0.0
    weighted_sum = 0.0
    for seg in segments:
        ov_start = max(clip_start, seg["start_sec"])
        ov_end = min(clip_end, seg["end_sec"])
        overlap = ov_end - ov_start
        if overlap > 0:
            weighted_sum += seg["value"] * overlap
            total_overlap += overlap
    if total_overlap <= 0:
        return None
    return weighted_sum / total_overlap

def _print_distribution(bins: dict, total: int) -> None:
    """점수 분포 히스토그램 출력 (0.0~1.0, 0.1 단위)"""
    
    logger.info("점수 분포 (0.1 단위):")
    for b in range(10):
        count = bins.get(b, 0)
        ratio = count / max(total, 1)
        bar = "#" * int(ratio * 80)
        logger.info(f"  {b / 10:.1f}~{(b + 1) / 10:.1f}: {count:5d} ({ratio * 100:4.1f}%) {bar}")

def relabel(input_path: Path, heatmap_path: Path, output_path: Path) -> dict:
    """combined 데이터셋을 회귀 라벨로 재생성"""
    
    heatmaps = load_heatmaps(heatmap_path)
    logger.info(f"히트맵 로드: {len(heatmaps)}개 영상")

    stats = {"total": 0, "relabeled": 0, "no_heatmap": 0, "no_clip": 0}
    bins: dict = defaultdict(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = rec.get("metadata", {})
            vid = meta.get("video_id")
            clip_start = meta.get("clip_start")
            clip_end = meta.get("clip_end")

            if vid not in heatmaps:
                stats["no_heatmap"] += 1
                continue
            if clip_start is None or clip_end is None:
                stats["no_clip"] += 1
                continue

            score = clip_engagement(heatmaps[vid], clip_start, clip_end)
            if score is None:
                stats["no_clip"] += 1
                continue

            score = round(score, 2)
            bins[min(int(score * 10), 9)] += 1

            rec["instruction"] = REGRESSION_INSTRUCTION
            rec["output"] = json.dumps({"engagement_score": score}, ensure_ascii=False)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["relabeled"] += 1
    
    logger.info(
        f"재라벨링 완료 | 총: {stats['total']} | 성공: {stats['relabeled']} "
        f"| 히트맵없음: {stats['no_heatmap']} | 클립정보없음: {stats['no_clip']}"
    )
    _print_distribution(bins, stats["relabeled"])
    return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="회귀 재라벨링 (32일차 신호 검증용)")
    parser.add_argument("--input", default="data/finetune/dataset_generator_p2_combined.jsonl")
    parser.add_argument("--heatmap", default="data/heatmaps/heatmaps_merged.jsonl")
    parser.add_argument("--output", default="data/finetune/dataset_regression_p1.jsonl")
    args = parser.parse_args()

    relabel(Path(args.input), Path(args.heatmap), Path(args.output))

if __name__ == "__main__":
    main()