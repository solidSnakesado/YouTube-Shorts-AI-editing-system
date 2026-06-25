# 계층: 진입점 (CLI 러너)
# 역할: GemmaNegBuilder 실행 진입점 (테스트 --max-videos 5 / 전체 네거티브 빌드 공용)
# 의존: gemma_neg_builder(C-3)
# 39일차 신규: Gemma 네거티브(비피크) 데이터셋 빌드 진입점. 빌더에 __main__ 없어 별도 러너.
#   - 반드시 포지티브 빌드(run_gemma_build.py) 후 실행 (dataset.jsonl의 영상당 수에 1:1 맞춤)
#   - 소요 시간/영상당 평균 출력 -> 전체 빌드 시간 외삽용

"""Gemma 네거티브 빌더 CLI 러너 - 비피크 구간 빈 하이라이트 샘플 생성 (포지티브와 1:1)"""

import argparse
import asyncio
import time
from pathlib import Path

from loguru import logger

from app.services.gemma_neg_builder import GemmaNegBuilder


async def _run(args: argparse.Namespace) -> None:
    """빌더 인스턴스화 -> build_negatives() 실행 -> 소요 시간/통계 출력"""

    builder = GemmaNegBuilder(
        heatmap_path=Path(args.heatmap),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_videos=args.max_videos,
    )
    started = time.monotonic()
    output_path = await builder.build_negatives(
        pos_filename=args.pos_filename,
        output_filename=args.output_filename,
        neg_per_video=args.neg_per_video,
        seed=args.seed,
    )
    elapsed = time.monotonic() - started

    stats = builder.stats
    logger.info(f"네거티브 빌드 종료 | 출력: {output_path}")
    logger.info(f"통계: {stats}")
    logger.info(f"소요: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    done = stats["processed"]
    if done:
        logger.info(f"영상당 평균: {elapsed / done:.1f}초")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 네거티브 데이터셋 빌더 러너")
    parser.add_argument(
        "--heatmap", default="data/heatmaps/heatmaps_merged.jsonl",
        help="히트맵 JSONL 경로 (기본: merged)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="처리할 최대 영상 수 (테스트용, 미지정 시 포지티브 보유 영상 전체)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="출력 루트 (기본: datasets/gemma_audio, 포지티브와 동일해야 함)",
    )
    parser.add_argument(
        "--pos-filename", default="dataset.jsonl",
        help="기준 포지티브 JSONL 파일명 (영상당 수 집계 대상)",
    )
    parser.add_argument(
        "--output-filename", default="dataset_neg.jsonl",
        help="네거티브 출력 JSONL 파일명 (포지티브와 분리)",
    )
    parser.add_argument(
        "--neg-per-video", type=int, default=0,
        help="영상당 네거티브 수 (0=포지티브 수에 1:1, >0=고정 수)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="비피크 윈도우 랜덤 선택 시드 (재현성)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()