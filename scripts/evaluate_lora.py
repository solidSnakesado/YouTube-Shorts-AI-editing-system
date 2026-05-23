# 계층: 스크립트 (CLI 진입점)
# 역할: LoRA 파인튜닝 전/후 하이라이트 추출 결과 비교 평가
# 의존: config.py, vlm_client.py, llm_highlight_extractor.py
# 23일차 신규: 동일 영상에서 기본 모델 vs LoRA 모델 비교
#
# 실행: uv run python -m scripts.evaluate_lora --video <경로> --transcript <경로>
# 옵션: --heatmap <JSONL> --video-id <ID> --max-shorts 5 --output ./data/eval

"""
LoRA 파인튜닝 전/후 비교 평가 스크립트
동일 영상에서 기본 Qwen2.5-VL vs LoRA 어댑터 적용 모델의 하이라이트 추출 결과를 비교
비교 지표: hook_score 분포, 히트맵 피크 IoU 일치율, 구간 시간 분포
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings

# --------------------------------------------------------------
# 히트맵 피크 추출 (IoU 비교용)
# --------------------------------------------------------------

def load_heatmap_peaks(heatmap_path: Path, video_id: str) -> list[dict]:
    """히트맵 JSONL에서 특정 영상의 피크 구간 추출"""

    if not heatmap_path.is_file():
        return []
    
    with open(heatmap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("video_id") == video_id:
                return _extract_peaks(record)
            
    return []

def _extract_peaks(record: dict) -> list[dict]:
    """히트맵 데이터에서 상위 피크 구간 추출 (상위 20% 임계값)"""

    heatmap = record.get("heatmap", [])
    if not heatmap:
        return []
    
    values = [pt.get("value", 0) for pt in heatmap]
    if not values:
        return []
    
    threshold = sorted(values, reverse=True)[max(0, len(values) // 5)]
    duration = record.get("duration_sec", 0)

    peaks = []
    in_peak = False
    peak_start = 0.0

    for pt in heatmap:
        time_sec = pt.get("start_sec", 0)
        val = pt.get("value", 0)

        if val >= threshold and not in_peak:
            in_peak = True
            peak_start = time_sec
        elif val < threshold and in_peak:
            in_peak = False
            peaks.append({"start_sec": peak_start, "end_sec": time_sec})

    if in_peak:
        peaks.append({"start_sec": peak_start, "end_sec": duration})

    return peaks

# --------------------------------------------------------------
# IoU 계산
# --------------------------------------------------------------

def compute_iou(seg_a: dict, seg_b: dict) -> float:
    """두 시간 구간의 IoU (Intersection over Union) 계산"""

    start = max(seg_a["start_sec"], seg_b["start_sec"])
    end = min(seg_a["end_sec"], seg_b["end_sec"])
    intersection = max(0.0, end - start)

    dur_a = seg_a["end_sec"] - seg_a["start_sec"]
    dur_b = seg_b["end_sec"] - seg_b["start_sec"]
    union = dur_a + dur_b - intersection

    return intersection / union if union > 0 else 0.0

def compute_peak_match_rate(highlights: list[dict], peaks: list[dict], iou_threshold: float = 0.3) -> dict:
    """하이라이트와 히트맵 피크 간 매칭률 계산"""

    if not highlights or not peaks:
        return {"match_rate": 0.0, "matched": 0, "total": len(highlights), "avg_iou": 0.0}
    
    matched = 0
    ious = []

    for hl in highlights:
        best_iou = max(compute_iou(hl, peak) for peak in peaks)
        ious.append(best_iou)
        if best_iou >= iou_threshold:
            matched += 1

    return {
        "match_rate": matched / len(highlights) if highlights else 0.0,
        "matched": matched,
        "total": len(highlights),
        "avg_iou": statistics.mean(ious) if ious else 0.0,
        "max_iou": max(ious) if ious else 0.0,
    }

# --------------------------------------------------------------
# 하이라이트 통계
# --------------------------------------------------------------

def compute_highlight_stats(highlights: list[dict]) -> dict:
    """하이라이트 리스트에서 통계 지표 산출"""

    if not highlights:
        return {"count": 0}
    
    scores = [h.get("hook_score", 0) for h in highlights]
    durations = [h.get("end_sec", 0) - h.get("start_sec", 0) for h in highlights]

    return {
        "count": len(highlights),
        "hook_score_mean": round(statistics.mean(scores), 4),
        "hook_score_stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "hook_score_max": round(max(scores), 4),
        "hook_score_min": round(min(scores), 4),
        "duration_mean": round(statistics.mean(durations), 1),
        "duration_min": round(min(durations), 1),
        "duration_max": round(max(durations), 1),
        "highlights": highlights,
    }

# --------------------------------------------------------------
# 평가 메인
# --------------------------------------------------------------

async def run_evaluation(
    video_path: str,
    transcript_path: str,
    heatmap_path: Optional[str] = None,
    video_id: Optional[str] = None,
    max_shorts: int = 5,
    output_dir: Optional[str] = None,
) -> dict:
    """파인튜닝 전/후 비교 평가 실행"""

    from app.services.vlm_client import run_vlm_analysis, is_vlm_available

    # 전사 데이터 로드
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    logger.info(f"영상: {video_path}")
    logger.info(f"전사 세그먼트: {len(transcript_data.get('segments', []))}개")

    # 1단계: 기본 모델 (LoRA 없이)
    logger.info("=" * 50)
    logger.info("[1/2] 기본 모델 (LoRA OFF) 하이라이트 추출")
    logger.info("=" * 50)

    if not is_vlm_available():
        raise RuntimeError("llama-server 또는 GGUF/mmproj 파일이 없습니다. 먼저 설치하세요.")
    
    base_highlights = await run_vlm_analysis(video_path, transcript_data, max_shorts)
    base_stats = compute_highlight_stats(base_highlights)
    logger.info(f"기본 모델: {base_stats['count']}개 하이라이트, 평균 hook_score={base_stats.get('hook_score_mean', 0)}")

    # 2단계: LoRA 모델
    logger.info("=" * 50)
    logger.info("[2/2] LoRA 모델 (LoRA ON) 하이라이트 추출")
    logger.info("=" * 50)

    adapter_path = str(settings.lora_adapter_path)
    if not Path(adapter_path).exists():
        raise RuntimeError(f"LoRA 어댑터 없음: {adapter_path}\n먼저 train_qlora.py로 학습하세요.")
    
    lora_highlights = await run_vlm_analysis(video_path, transcript_data, max_shorts, lora_adapter_path=adapter_path)
    lora_stats = compute_highlight_stats(lora_highlights)
    logger.info(f"LoRA 모델: {lora_stats['count']}개 하이라이트, 평균 hook_score={lora_stats.get('hook_score_mean', 0)}")

    # 3단계: 비교
    result = {
        "video_path": video_path,
        "transcript_path": transcript_path,
        "max_shorts": max_shorts,
        "base_model": base_stats,
        "lora_model": lora_stats,
    }

    # 히트맵 피크 비교 (데이터 제공 시)
    if heatmap_path and video_id:
        peaks = load_heatmap_peaks(Path(heatmap_path), video_id)
        if peaks:
            logger.info(f"히트맵 피크: {len(peaks)}개 구간")
            result["heatmap_peaks"] = len(peaks)
            result["base_peak_match"] = compute_peak_match_rate(base_highlights, peaks)
            result["lora_peak_match"] = compute_peak_match_rate(lora_highlights, peaks)

    # 결과 저장
    out_dir = Path(output_dir) if output_dir else Path("./data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "eval_result.json"

    # highlights 내부의 상세 데이터를 직렬화 가능하게 정리
    serializable = json.loads(json.dumps(result, default=str))
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    _print_comparison(result)
    logger.info(f"평가 결과 저장: {result_path}")
    return result

def _print_comparison(result: dict) -> None:
    """비교 결과를 콘솔에 요약 출력"""

    base = result["base_model"]
    lora = result["lora_model"]

    print("\n" + "=" * 60)
    print("  파인튜닝 전/후 비교 결과")
    print("=" * 60)

    print(f"\n{'지표':<25} {'기본 모델':>12} {'LoRA 모델':>12} {'변화':>10}")
    print("-" * 60)

    for key in ["count", "hook_score_mean", "hook_score_stdev", "duration_mean"]:
        bv = base.get(key, 0)
        lv = lora.get(key, 0)
        diff = lv - bv if isinstance(bv, (int, float)) else ""
        sign = "+" if isinstance(diff, (int, float)) and diff > 0 else ""
        print(f"    {key:<23} {bv:>12} {lv:>12} {sign}{diff:>9}")

    if "base_peak_match" in result:
        print(f"\n{'히트맵 일치율':<25} {'기본 모델':>12} {'LoRA 모델':>12}")
        print("-" * 60)
        bm = result["base_peak_match"]
        lm = result["lora_peak_match"]
        print(f"    {'매칭률':<23} {bm['match_rate']:>11.1%} {lm['match_rate']:>11.1%}")
        print(f"    {'평균 IoU':<23} {bm['avg_iou']:>12.3f} {lm['avg_iou']:>12.3f}")
        print(f"    {'최대 IoU':<23} {bm['max_iou']:>12.3f} {lm['max_iou']:>12.3f}")

    print("=" * 60 + "\n")

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 파인튜닝 전/후 비교 평가")
    parser.add_argument("--video", type=str, required=True, help="평가할 영상 파일 경로")
    parser.add_argument("--transcript", type=str, required=True, help="전사 결과 JSON 파일 경로")
    parser.add_argument("--heatmap", type=str, default=None, help="히트맵 JSONL 경로 (IoU 비교용)")
    parser.add_argument("--video-id", type=str, default=None, help="히트맵에서 찾을 영상 ID")
    parser.add_argument("--max-shorts", type=int, default=5, help="추출할 최대 쇼츠 수 (기본: 5)")
    parser.add_argument("--output", type=str, default=None, help="평가 결과 저장 디렉토리")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not Path(args.video).is_file():
        raise FileNotFoundError(f"영상 파일 없음: {args.video}")
    if not Path(args.transcript).is_file():
        raise FileNotFoundError(f"전사 파일 없음: {args.transcript}")
    
    asyncio.run(run_evaluation(
        video_path=args.video,
        transcript_path=args.transcript,
        heatmap_path=args.heatmap,
        video_id=args.video_id,
        max_shorts=args.max_shorts,
        output_dir=args.output,
    ))

if __name__ == "__main__":
    main()