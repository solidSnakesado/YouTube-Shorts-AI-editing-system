# 계층: 비즈니스 로직 계층 (Service)
# 역할: YouTube "Most Replayed" 히트맵 데이터를 yt-dlp로 수집하여 JSONL 저장
#      공식 Data API에 히트맵이 없으므로 yt-dlp info.heatmap(InnerTube 파싱) 사용
# 의존: config.py, yt-dlp (외부 CLI)
# 20일차 신규: 파인튜닝 파이프라인 1단계 (데이터 수집)
#             21일차 프레임 + 히트맵 페어 -> 22일차 QLoRA 파인튜닝 라벨로 사용
# 46일차 수정(1회): 중복 수집 스킵 추가. 이미 수집된 video_id면 yt-dlp 호출 없이 즉시 스킵.
#   반복 증분 수집 시 같은 영상 재다운로드 -> JSONL 중복 라인 + yt-dlp 재실행(시간/403 비용)
#   해소. output_dir의 모든 *.jsonl 스캔으로 기수집 ID 집합 구성(지연 로드 + 인스턴스 캐시).
#   변경: __init__ 캐시(L42), collect_single 중복 체크(L56-62) + 성공 시 캐시 갱신(L103),
#         load_collected_ids() 신규(L112-135). 다운스트림 빌더 dedup과 별개로 수집 시점 차단.

"""
히트맵 수집 서비스 - YouTube "Most Replayed" 히트맵을 yt-dlp로 추출하여 JSONL 저장
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings

# 상수
PEAK_THRESHOLD: float = 0.8             # 피크 판별 기준 (이 값 이상 = 시청자가 많이 다시 본 구간)
PEAK_MIN_DURATION_SEC: float = 5.0      # 피크 최소 길이 (초), 짧은 피크는 노이즈

# YouTube URL -> video_id 추출 정규식
_VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})"
)

class HeatmapCollector:
    """YouTube 히트맵 수집기 - yt-dlp로 히트맵 추출 -> 정규화 -> 피크 식별 -> JSONL 저장"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.heatmap_output_path
        self._known_ids: Optional[set] = None   # 46일차: 기수집 video_id 캐시 (중복 스킵, 지연 로드)

    # --------------------------------------------------------------
    # 공개 API
    # --------------------------------------------------------------

    async def collect_single(self, url: str) -> Optional[dict]:
        """단일 영상의 히트맵을 수집한다. 실패/중복 시 None 반환"""

        video_id = self._extract_video_id(url)
        if not video_id:
            logger.warning(f"유효하지 않은 YouTube URL: {url}")
            return None

        # 46일차: 중복 스킵 - 이미 수집된 영상이면 yt-dlp 호출 없이 즉시 종료
        #   (재수집 시 JSONL 중복 라인 + 메타데이터 재조회 비용/403 위험 제거)
        if self._known_ids is None:
            self._known_ids = self.load_collected_ids()
        if video_id in self._known_ids:
            logger.info(f"이미 수집됨, 스킵: {video_id}")
            return None

        logger.info(f"히트맵 수집 시작: {video_id}")

        # yt-dlp로 메타데이터 추출 (영상 다운로드 없이 JSON만)
        raw_info = await self._fetch_metadata(url)
        if not raw_info:
            return None

        # 영상 길이 검증
        duration = raw_info.get("duration", 0)
        if duration < settings.HEATMAP_MIN_DURATION_SEC:
            logger.warning(
                f"영상 너무 짧아 스킵: {video_id} "
                f"({duration:.0f}초 < {settings.HEATMAP_MIN_DURATION_SEC:.0f}초)"
            )
            return None

        # 히트맵 데이터 추출 및 검증
        raw_heatmap = raw_info.get("heatmap")
        if not raw_heatmap:
            logger.warning(f"히트맵 없음 (비공개 또는 조회수 부족): {video_id}")
            return None

        heatmap = self._normalize_heatmap(raw_heatmap, duration)
        if not heatmap:
            logger.warning(f"히트맵 정규화 실패: {video_id}")
            return None

        # 피크 세그먼트 식별
        peaks = self._find_peak_segments(heatmap)

        result = {
            "video_id": video_id,
            "title": raw_info.get("title", ""),
            "duration_sec": round(duration, 2),
            "heatmap": heatmap,
            "peak_segments": peaks,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        self._known_ids.add(video_id)   # 46일차: 같은 배치 내 후속 중복도 잡도록 캐시 갱신

        logger.info(
            f"히트맵 수집 완료: {video_id} | "
            f"구간={len(heatmap)}개, 피크={len(peaks)}개"
        )

        return result

    def load_collected_ids(self) -> set:
        """output_dir의 모든 *.jsonl을 스캔해 이미 수집된 video_id 집합을 반환한다 (중복 스킵용).
        46일차 신규. 날짜별 파일(heatmaps_YYYY-MM-DD.jsonl) + merged 전체를 대상으로 한다."""

        ids: set = set()
        if not self.output_dir.is_dir():
            return ids
        for filepath in self.output_dir.glob("*.jsonl"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            vid = json.loads(line).get("video_id")
                        except json.JSONDecodeError:
                            continue
                        if vid:
                            ids.add(vid)
            except OSError as e:
                logger.warning(f"기수집 ID 스캔 실패: {filepath.name} | {e}")
        logger.debug(f"기수집 video_id {len(ids)}개 로드 (중복 스킵 기준)")
        return ids

    def append_to_jsonl(self, data: dict, filename: str) -> Path:
        """수집 결과를 JSONL 파일에 1라인 추가한다"""

        filepath = self.output_dir / filename

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.debug(f"JSONL 추가: {filepath} | video_id={data['video_id']}")

        return filepath

    # --------------------------------------------------------------
    # 내부 메서드
    # --------------------------------------------------------------

    async def _fetch_metadata(self, url: str) -> Optional[dict]:
        """yt-dlp --dump-json으로 메타데이터 추출 (영상 다운로드 없음)"""

        cmd = [
            "yt-dlp",
            "--skip-download",          # 영상 파일 다운로드 안 함
            "--no-playlist",            # 재생목록이면 단일 영상만
            "--dump-json",              # JSON 메타데이터만 stdout 출력
            "--no-warnings",            # 경고 메시지 숨김
            "--socket-timeout",         # 네트워크 타임아웃
            str(settings.HEATMAP_REQUEST_TIMEOUT_SEC),
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=settings.HEATMAP_REQUEST_TIMEOUT_SEC + 10,
            )

            if process.returncode != 0:
                logger.error(
                    f"yt-dlp 실패 (rc={process.returncode}): "
                    f"{stderr.decode(errors='replace')[:200]}"
                )

                return None

            return json.loads(stdout.decode("utf-8"))

        except asyncio.TimeoutError:
            logger.error(f"yt-dlp 타임아웃: {url}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"yt-dlp JSON 파싱 실패: {e}")
            return None

    @staticmethod
    def _normalize_heatmap(raw_heatmap: list[dict], duration: float) -> list[dict]:
        """yt-dlp 히트맵을 정규화: 키 변환, value/시간 클램핑, 시간순 정렬"""

        normalized = []

        for entry in raw_heatmap:
            start = float(entry.get("start_time", 0))
            end = float(entry.get("end_time", 0))
            value = float(entry.get("value", 0))

            # 시간 범위 클램핑
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))

            # 역전 방지
            if end <= start:
                continue

            # value 클램핑 (0~1)
            value = max(0.0, min(1.0, value))

            normalized.append({
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "value": round(value, 4),
            })

        # 시간순 정렬
        normalized.sort(key=lambda x: x["start_sec"])

        return normalized

    @staticmethod
    def _find_peak_segments(heatmap: list[dict]) -> list[dict]:
        """히트맵 에서 PEAK_THRESHOLD 이상 연속 구간을 피크 세그먼트로 병합"""

        if not heatmap:
            return []

        peaks = []
        current_start = None
        current_end = None
        value_sum = 0.0
        value_count = 0

        for entry in heatmap:
            if entry["value"] >= PEAK_THRESHOLD:
                if current_start is None:
                    # 새 피크 시작
                    current_start = entry["start_sec"]

                current_end = entry["end_sec"]
                value_sum += entry["value"]
                value_count += 1
            else:
                # 피크 종료 -> 저장
                if current_start is not None:
                    duration = current_end - current_start
                    if duration >= PEAK_MIN_DURATION_SEC:
                        peaks.append({
                            "start_sec": current_start,
                            "end_sec": current_end,
                            "avg_value": round(value_sum / value_count, 4),
                        })
                    current_start = None
                    value_sum = 0.0
                    value_count = 0

        # 마지막 피크 처리
        if current_start is not None:
            duration = current_end - current_start
            if duration >= PEAK_MIN_DURATION_SEC:
                peaks.append({
                    "start_sec": current_start,
                    "end_sec": current_end,
                    "avg_value": round(value_sum / value_count, 4),
                })

        return peaks

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        """YouTube URL에서 11자리 video_id 추출, 실패시 None"""

        match = _VIDEO_ID_RE.search(url)
        return match.group(1) if match else None