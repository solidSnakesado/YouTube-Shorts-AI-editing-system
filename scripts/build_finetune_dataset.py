# 계층: 스크립트 (CLI 진입점)
# 역할: 히트맵 JSONL에서 VLM 파인튜닝 데이터셋을 생성하는 CLI 래퍼
#       app/services/dataset_builder.py의 DatasetBuilder를 호출
# 의존: dataset_builder.py, config.py
# 21일차 신규: 파인튜닝 파이프라인 2단계 CLI
#
# 사용법:
#   # 기본 실행 (피크 2개 이상 영상 자동 선별)
#   uv run python -m scripts.build_finetune_dataset \
#       --heatmap data/heatmaps/heatmaps/heatmaps_2026-05-10.jsonl
#
#   # 최소 피크 수 지정
#       uv run python -m scripts.build_finetune_dataset \
#       --heatmap data/heatmaps/heatmaps/heatmaps_2026-05-10.jsonl
#       --min-peaks 3
#   
#   # 출력 파일명 지정
#   uv run python -m scripts.build_finetune_dataset \ 
#       --heatmap data/heatmaps/heatmaps/heatmaps_2026-05-10.jsonl
#       --output data/finetune/dataser.jsonl
#
# 출력: ./data/finetune/dataset.jsonl + ./data/finetune/frames/*.jpg

"""
파인튜닝 데이터셋 빌드 CLI - 히트맵 JSONL -> 프레임 추출 -> dataset.jsonl 생성
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# 프로젝트 루트를 sys.path에 추가 (스크립트 직접 실행 시 import 해결)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import settings
from app.services.dataset_builder import DatasetBuilder

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "   uv run python -m scripts.build_finetune_dataset \\\n"
            "       --heatmap data/heatmaps/heatmaps_2026-05-10.jsonl\n"
            "\n"
            "   uv run python -m scripts.build_finetune_dataset \\\n"
            "       --heatmap data/heatmaps/heatmaps_2026-05-10.jsonl \\\n"
            "       --min-peaks 3 --output data/finetune/ny_dataset.jsonl"
        ),
    )

    parser.add_argument(
        "--heatmap", required=True, type=Path,
        help="히트맵 JSONL 파일 경로 (heatmap_collector 산출물)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"출력 dataset.jsonl 경로 (기본: {settings.FINETUNE_OUTPUT_DIR}/dataset.jsonl)",
    )
    parser.add_argument(
        "--min-peaks", type=int, default=None,
        help=f"처리 대상 최소 피크 수 (기본: {settings.FINETUNE_MIN_PEAK_COUNT})",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help=f"세그먼트당 추출 프레임 수 (기본: {settings.FINETUNE_FRAMES_PER_SEGMENT})",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=None,
        help=f"네거티브/포지티브 비율 (기본: {settings.FINETUNE_NEGATIVE_RATIO})",
    )
    parser.add_argument(
        "--mode", type=str, default="classifier", choices=["classifier", "generator"],
        help="데이터셋 모드: classifier(판별기) 또는 generator(생성기) (기본: classifier)",
    )
    
    return parser.parse_args()

async def _run(args: argparse.Namespace) -> None:
    """메인 실행 로직"""

    # 입력 파일 검증
    if not args.heatmap.is_file():
        logger.error(f"히트맵 JSONL 파일을 찾을 수 없습니다: {args.heatmap}")
        sys.exit(1)

    # 출력 경로 결정
    output_dir = None
    output_filename = "dataset.jsonl"
    if args.output:
        output_dir = args.output.parent
        output_filename = args.output.name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # DatasetBuilder 생성
    builder = DatasetBuilder(
        heatmap_path=args.heatmap,
        output_dir=output_dir,
        min_peak_count=args.min_peaks,
        frames_per_segment=args.frames,
        negative_ratio=args.neg_ratio,
        mode=args.mode,
    )

    logger.info("=" * 60)
    logger.info("파인튜닝 데이터셋 빌드 시작")
    logger.info(f"  히트맵: {args.heatmap}")
    logger.info(f"  최소 피크: {builder.min_peak_count}개")
    logger.info(f"  프레임/세그먼트: {builder.frames_per_segment}장")
    logger.info(f"  네거티브 비율: {builder.negative_ratio}")
    logger.info(f"  모드: {builder.mode}")
    logger.info("=" * 60)

    # 빌드 실행
    result_path = await builder.build(output_filename=output_filename)

    # 결과 출력
    stats = builder.stats
    logger.info("=" * 60)
    logger.info("빌드 완료 요약")
    logger.info(f"  전체 영상: {stats['total_videos']}개")
    logger.info(f"  선별 영상: {stats['filtered_videos']}개")
    logger.info(f"  처리 성공: {stats['processed']}개")
    logger.info(f"  처리 실패: {stats['skipped']}개")
    logger.info(f"  포지티브 샘플: {stats['positive_samples']}개")
    logger.info(f"  네거티브 샘플: {stats['negative_samples']}개")
    total = stats['positive_samples'] + stats['negative_samples']
    logger.info(f"  총 샘플: {total}개")
    logger.info(f"  출력 파일: {result_path}")
    logger.info("=" * 60)

def main():
    """CLI 진입점"""

    args = _parse_args()
    asyncio.run(_run(args))

if __name__ == "__main__":
    main()
