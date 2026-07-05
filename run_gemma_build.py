# 계층: 진입점 (CLI 러너)
# 역할: GemmaDatasetBuilder 실행 진입점 (테스트 --max-videos 5 / 전체 빌드 공용)
# 의존: gemma_dataset_builder(C-2b)
# 39일차 신규: Gemma 오디오 피벗 데이터셋 빌드 진입점. 빌더에 __main__ 없어 별도 러너.
#   - 소요 시간/영상당 평균 출력 -> 전체 빌드 시간 외삽용

"""Gemma 데이터셋 빌더 CLI 러버 - 테스트(--max-videos 5) 및 전체 빌드 공용"""

import argparse
import asyncio
import time
from pathlib import Path

from loguru import logger

from app.services.gemma_dataset_builder import GemmaDatasetBuilder

async def _run(args: argparse.Namespace) -> None:
    """빌더 인스턴스화 -> build() 실행 -> 소요 시간/통계 출력"""

    builder = GemmaDatasetBuilder(
        heatmap_path=Path(args.heatmap),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        min_peak_count=args.min_peak_count,
        max_videos=args.max_videos,
        delay=args.delay,
    )
    started = time.monotonic()
    output_path = await builder.build()
    elapsed = time.monotonic() - started

    stats = builder.stats
    logger.info(f"빌드 종료 | 출력: {output_path}")
    logger.info(f"통계: {stats}")
    logger.info(f"소요: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    done = stats["processed"]
    if done:
        logger.info(f"영상당 평균: {elapsed / done:.1f}초")

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 오디오 데이터셋 빌더 러너")
    parser.add_argument(
        "--heatmap", default="data/heatmaps/heatmaps_merged.jsonl",
        help="히트맵 JSONL 경로 (기본: merged)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="처리할 최대 영상 수 (테스트용, 미지정 시 전체)",
    )
    parser.add_argument(
        "--min-peak-count", type=int, default=2,
        help="최소 피크 수 (기본 2, Qwen 베이스라인 일치)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="출력 루트 (기본: datasets/gemma_audio)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="영상 간 대기(초) - 403 rate limit 방어. 권장 3~5",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))

if __name__ == "__main__":
    main()