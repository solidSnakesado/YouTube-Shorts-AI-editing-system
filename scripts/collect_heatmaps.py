# 계층: 스크립트 (CLI 진입점)
# 역할: 다수 YouTube URL의 "Most Replayed" 히트맵을 일괄 수집하여 JSONL로 저장
#       app/services/heatmap_collector.py의 HeatmapCollector를 호출하는 래퍼
# 의존: heatmap_collector.py, config.py
# 20일차 신규: 파인튜닝 파이프라인 1단계 CLI
#
# 사용법:
#   # URL을 직접 나열
#   uv run python -m scripts.collect_heatmaps \
#       "https://www.youtube.com/watch?v=abc" \
#       "https://www.youtube.com/watch?v=def"
#
#   # 파일에서 URL 목록 읽기 (한 줄에 하나)
#   uv run python -m scripts.collect_heatmaps --file urls.txt
#   
#   # 출력 파일명 지정
#   uv run python -m scripts.collect_heatmaps --file urls.txt --output my_heatmaps.jsonl
#
# 출력: ./data/heatmaps/{output}.jsonl (config.HEATMAP_OUTPUT_DIR)

"""
히트맵 일괄 수집 CLI - YouTube URL 목록에서 히트맵을 수집하여 JSONL 저장
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# 프로젝트 루트를 sys.path에 추가 (스크립트 직접 실행 시 import 해결)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import settings
from app.services.heatmap_collector import HeatmapCollector

def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""

    parser = argparse.ArgumentParser(description="YouTube 히트맵 일괄 수집 (파인튜닝 데이터용)")
    parser.add_argument("urls", nargs="*", default=[], help="수집할 YouTube URL (여러 개 나열 가능)")
    parser.add_argument("-f", "--file", type=str, default=None, help="URL 목록 파일 경로 (한 줄에 URL 하나)")
    parser.add_argument("-o", "--output", type=str, default=None, help="JSONL 출력 파일명 (기본: heatmaps_YYYY-MM-DD.jsonl)")
    parser.add_argument("--rate-limit", type=float, default=None, help=f"영상 간 대기 시간(초) (기본: {settings.HEATMAP_RATE_LIMIT_SEC})")
    parser.add_argument("--max-retry", type=int, default=2, help="실패 시 재시도 횟수 (기본: 2)")

    return parser.parse_args()

def load_urls(args: argparse.Namespace) -> list[str]:
    """CLI 인자와 파일에서 URL 목록을 합친다"""

    urls = list(args.urls)

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            logger.error(f"URL 파일 없음: {filepath}")
            sys.exit(1)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 빈 줄, 주석(#) 무시
                if line and not line.startswith("#"):
                    urls.append(line)
    
    # 중복 제거 (순서 유지)
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    
    return unique

async def run_collection(urls: list[str], output_filename: str, rate_limit_sec: float, max_retry: int) -> dict:
    """
    URL 목로에서 순회하며 히트맵을 수집한다.

    Returns:
        {"success": int, "skipped": int, "failed": int, "total": int}
    """

    collector = HeatmapCollector()
    stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(urls)}

    for idx, url in enumerate(urls, 1):
        logger.info(f"[{idx}/{len(urls)}] 처리 중: {url}")

        result = None
        last_error = None

        # 재시도 루프
        for attempt in range(1, max_retry + 1):
            try:
                result = await collector.collect_single(url)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"   시도 {attempt}/{max_retry} 실패: {e}")

                if attempt < max_retry:
                    await asyncio.sleep(rate_limit_sec)
        
        if result:
            collector.append_to_jsonl(result, output_filename)
            stats["success"] += 1
            logger.info(
                f"  수집 완료: {result['video_id']} | "
                f"구간={len(result['heatmap'])}개, "
                f"피크={len(result['peak_segments'])}개"
            )
        elif result is None and last_error is None:
            # collect_single이 None 반환 (히트맵 없음/짧은 영상 등)
            stats["skipped"] += 1
            logger.info("   - 스킵 (히트맵 없음 또는 조건 미충족)")
        else:
            stats["failed"] += 1
            logger.error(f"     x 최종 실패: {last_error}")

        # 마지막 URL이 아니면 rate_limit 대기
        if idx < len(urls):
            await asyncio.sleep(rate_limit_sec)

    return stats

def main():
    """CLI 메인 진입점"""

    args = parse_args()
    urls = load_urls(args)

    if not urls:
        logger.error("수집할 URL이 없습니다. URL을 인자로 전달하거나 --file 옵션을 사용하세요.")
        sys.exit(1)

    # 출력 파일명 결정
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_filename = args.output or f"heatmaps_{today}.jsonl"

    # rate limit 결정
    rate_limit = args.rate_limit or settings.HEATMAP_RATE_LIMIT_SEC

    logger.info(f"히트맵 수집 시작: {len(urls)}개 URL")
    logger.info(f"출력 파일: {settings.heatmap_output_path / output_filename}")
    logger.info(f"Rate limit: {rate_limit}초, 최대 재시도: {args.max_retry}회")

    # 비동기 수집 실행
    stats = asyncio.run(run_collection(urls, output_filename, rate_limit, args.max_retry))

    # 결과 요약
    logger.info(
        f"수집 완료 | "
        f"성공: {stats['success']}, "
        f"스킵: {stats['skipped']}, "
        f"실패: {stats['failed']}, "
        f"전체: {stats['total']}, "
    )

    # 실패가 과반이면 종료 코드 1
    if stats["failed"] > stats["total"] // 2:
        logger.error("과반 이상 실패 - 네트워크 또는 yt-dlp 문제를 확인하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()