# 계층: 스크립트 (CLI 진입점)
# 역할: 네거티브 데이터셋 빌드 CLI - 기존 포지티브 JSONL 기반 비피크 구간 수집
# 의존: dataset_neg_builder.py
# 31일차 신규: 네거티브 샘플 수집 전용 CLI
#
# 사용법:
#   # 기본 실행 (포지티브 수와 1:1 비율)
#   uv run python -m scripts.build_neg_dataset \
#       --heatmap data/heatmaps/heatmaps_merged.jsonl \
#       --positive data/finetune/dataset_generator_p2.jsonl \
#       --output data/finetune/dataset_generator_p2_neg.jsonl
#
#   # 영상당 네거티브 수 직접 지정
#       uv run python -m scripts.build_neg_dataset \
#       --heatmap data/heatmaps/heatmaps_merged.jsonl
#       --positive data/finetune/dataset_generator_p2.jsonl \
#       --output data/finetune/dataset_generator_p2_neg.jsonl \ 
#       --neg-per-video 5

"""네거티브 데이터셋 빌드 CLI - Phase 2 생성기 학습용 비피크 구간 수집"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.dataset_neg_builder import build_negative_dataset

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="네거티브 데이터셋 빌드 (Phase 2 생성기)")
    parser.add_argument(
        "--heatmap", required=True, type=Path,
        help="히트맵 JSONL 파일 경로"
    )
    parser.add_argument(
        "--positive", required=True, type=Path,
        help="기존 포지티브 JSONL 파일 경로 (처리된 영상 참조)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="네거티브 JSONL 출력 경로",
    )
    parser.add_argument(
        "--neg-per-video", type=int, default=0,
        help="영상당 네거티브 클립 수 (0=포지티브 수에 맞춤, 기본: 0)",
    )
    return parser.parse_args()

async def _run(args: argparse.Namespace) -> None:
    if not args.heatmap.is_file():
        logger.error(f"히트맵 파일 없음: {args.heatmap}")
        sys.exit(1)
    if not args.positive.is_file():
        logger.error(f"포지티브 JSONL 없음: {args.positive}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("네거티브 데이터셋 빌드 시작")
    logger.info(f"  히트맵: {args.heatmap}")
    logger.info(f"  포지티브: {args.positive}")
    logger.info(f"  출력: {args.output}")
    logger.info(f"  영상당 네거티브: {'포지티브 수에 맞춤' if args.neg_per_video == 0 else f'{args.neg_per_video}개'}")
    logger.info("=" * 60)

    stats = await build_negative_dataset(
        heatmap_path=args.heatmap,
        pos_jsonl_path=args.positive,
        output_path=args.output,
        neg_per_video=args.neg_per_video,
    )

    logger.info("=" * 60)
    logger.info("네거티브 빌드 완료 요약")
    logger.info(f"  전체 영상: {stats['total']}개")
    logger.info(f"  처리 성공: {stats['processed']}개")
    logger.info(f"  처리 실패: {stats['skipped']}개")
    logger.info(f"  네거티브 샘플: {stats['samples']}개")
    logger.info(f"  출력 파일: {args.output}")
    logger.info("=" * 60)

def main():
    args = _parse_args()
    asyncio.run(_run(args))

if __name__ == "__main__":
    main()