# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 히트맵 비피크 구간 -> 30s 구간 다운로드 -> [1fps 프레임 + 오디오] -> 빈 하이라이트(네거티브) 샘플 생성
#       Gemma 4 E4B 오디오 피벗 데이터 재구축 - 모듈 C-3 (네거티브 빌더)
# 의존: gemma_dataset_builder.GemmaDatasetBuilder(상속: 경로/추출/저장/정규화 재사용),
#       gemma_ytdlp.download_video_section(C-1), gemma_sample(C-2a 공유 상수), dataset_utils, gemma_config
# 39일차 신규: all-positive 문제 해소 -> 비피크 구간을 빈 하이라이트로 학습(빈 출력 + 오디오 기반 판별 강제).
#   - 포지티브 수에 맞춰 1:1 (영상당 포지티브 샘플 수만큼 네거티브 생성)
#   - 별도 파일(dataset_neg.jsonl) + 독립 재개 -> 포지티브 빌드와 무관하게 이어서 진행
#   - 파일명은 neg_id(f"{vid}_neg")로 격리 -> 포지티브 프레임/오디오와 충돌 방지

"""Gemma 네거티브 빌더 - 비피크 30s 구간에서 빈 하이라이트 샘플 생성 (포지티브와 1:1)"""

import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.gemma_config import gemma_settings
from app.services.dataset_utils import load_processed_ids
from app.services.gemma_ytdlp import download_video_section
from app.services.gemma_sample import build_gemma_sample, HIGHLIGHT_INSTRUCTION, NEGATIVE_OUTPUT
from app.services.gemma_dataset_builder import GemmaDatasetBuilder

class GemmaNegBuilder(GemmaDatasetBuilder):
    """비피크 30s 구간 -> 빈 하이라이트(네거티브) 샘플. GemmaDatasetBuilder 상속(경로/추출/저장 재사용)"""

    async def build_negatives(
        self,
        pos_filename: str = "dataset.jsonl",
        output_filename: str = "dataset_neg.jsonl",
        neg_per_video: int = 0,
        seed: int = 42,
    ) -> Path:
        """포지티브 수에 맞춰(1:1) 비피크 네거티브 생성 (재개 가능). neg_per_video>0이면 영상당 고정 수"""

        random.seed(seed)
        pos_path = self.output_dir / pos_filename
        output_path = self.output_dir / output_filename
        pos_counts = self._count_pos_per_video(pos_path)        # 39일차: 영상당 포지티브 수 (1:1 기준)
        if not pos_counts:
            logger.error(f"포지티브 JSONL에 영상 없음 (먼저 dataset.jsonl 생성 필요): {pos_path}")
            return output_path
        neg_done = load_processed_ids(output_path)              # 네거티브 video_id 기준 독립 재개
        if neg_done:
            logger.info(f"이어서 진행 | 네거티브 기처리 {len(neg_done)}개 스킵")
        all_records = self._load_all()
        # 포지티브가 있는 영상만 대상 (피크 필터는 포지티브 단계에서 이미 적용됨)
        videos = [r for r in all_records if r.get("video_id") in pos_counts]
        if self.max_videos:
            videos = videos[: self.max_videos]
        self._stats["total_videos"] = len(all_records)
        self._stats["filtered_videos"] = len(videos)
        logger.info(f"Gemma 네거티브 빌드 시작 | 대상 {len(videos)}개 (포지티브 보유 영상)")
        for idx, video in enumerate(videos, 1):
            vid = video["video_id"]
            if vid in neg_done:
                self._stats["processed"] += 1
                continue
            count = neg_per_video if neg_per_video > 0 else pos_counts.get(vid, 0)
            if count <= 0:
                continue
            logger.info(f"[{idx}/{len(videos)}] 네거티브: {vid} | 목표 {count}개")
            try:
                samples = await self._process_video_negatives(video, count)
                self._append_samples(output_path, samples)
                self._stats["processed"] += 1
                logger.info(f"[{idx}/{len(videos)}] 완료: {vid} | 네거티브 {len(samples)}개")
            except Exception as e:
                logger.error(f"[{idx}/{len(videos)}] 실패: {vid} | {e}")
                self._stats["skipped"] += 1
        logger.info(
            f"Gemma 네거티브 빌드 완료 | 처리 {self._stats['processed']}, 스킵 {self._stats['skipped']} | "
            f"샘플 {self._stats['samples']}개, 오디오없음 스킵 {self._stats['no_audio_skips']}개, "
            f"구간 다운로드 실패 {self._stats['peak_dl_fails']}개"
        )
        return output_path

    async def _process_video_negatives(self, video: dict, count: int) -> list[dict]:
        """비피크 30s 윈도우 count개 선택 -> 각각 구간 다운로드 -> [프레임+오디오] -> 빈 레이블. 구간별 실패 격리"""

        vid = video["video_id"]
        duration = video.get("duration_sec", 0.0)
        title = video.get("title", "")
        peaks = self._normalize_peaks(video.get("peak_segments", []))   # 피크 영역 회피용(상속 재사용)
        clips = self._pick_negative_clips(peaks, duration, count)
        if not clips:
            return []
        temp_dir = Path("temp") / "gemma"
        temp_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict] = []
        for ci, (clip_start, clip_end) in enumerate(clips, 1):
            section_path = temp_dir / f"{vid}_{int(clip_start)}_neg.mp4"
            try:
                # C-1: 비피크 구간만 다운로드 (절대 시각). 출력 파일은 0초부터 시작.
                await download_video_section(vid, clip_start, clip_end, section_path)
                if not section_path.is_file():
                    raise RuntimeError("구간 다운로드 산출물 없음")
                sample = await self._build_negative_clip(
                    section_path, vid, title, duration, clip_start, clip_end
                )
                if sample:
                    samples.append(sample)
            except Exception as e:
                self._stats["peak_dl_fails"] += 1           # 39일차: 구간별 실패 가시화(포지티브와 동일 카운터)
                logger.warning(
                    f"네거티브 구간 실패: {vid} [{ci}/{len(clips)}] {clip_start:.0f}s | {e}"
                )
            finally:
                if section_path.is_file():
                    section_path.unlink()
        return samples

    async def _build_negative_clip(
        self, section_path: Path, vid: str, title: str, duration: float,
        clip_start: float, clip_end: float,
    ) -> Optional[dict]:
        """0초 시작 구간 파일 전체(=네거티브 클립)에서 [프레임+오디오] -> 빈 하이라이트 샘플"""

        seg_len = clip_end - clip_start             # 구간 파일은 0-base, 파일 전체가 클립
        neg_id = f"{vid}_neg"                       # 39일차: 경로 격리(포지티브 프레임/오디오와 충돌 방지)
        # 오디오 먼저: 트랙 없으면 클립 스킵 (오디오 모델이므로 오디오 필수)
        audio_rel = await self._extract_audio(section_path, neg_id, clip_start, 0.0, seg_len)
        if audio_rel is None:
            self._stats["no_audio_skips"] += 1
            return None
        frames = await self._extract_frames(section_path, neg_id, clip_start, 0.0, seg_len)
        if not frames:
            return None
        # 39일차: 레이블=빈 하이라이트(NEGATIVE_OUTPUT). 메타에 label 마커(추적/집계용, 모델 입력 아님)
        meta = {
            "video_id": vid, "video_title": title, "duration_sec": duration,
            "clip_start": round(clip_start, 1), "clip_end": round(clip_end, 1),
            "label": "negative",
        }
        self._stats["samples"] += 1
        return build_gemma_sample(frames, audio_rel, HIGHLIGHT_INSTRUCTION, NEGATIVE_OUTPUT, meta)

    @staticmethod
    def _pick_negative_clips(
        peaks: list[dict], duration: float, count: int,
    ) -> list[tuple[float, float]]:
        """피크와 겹치지 않는 30s 윈도우를 랜덤 선택 (피크 ±5s 버퍼, 비중첩)"""

        win = float(gemma_settings.GEMMA_AUDIO_MAX_SEC)
        peak_secs: set[int] = set()
        for p in peaks:
            for s in range(max(0, int(p["start_sec"]) - 5), int(p["end_sec"]) + 6):
                peak_secs.add(s)
        candidates: list[float] = []
        t = 0.0
        while t + win <= duration:
            clip_secs = set(range(int(t), int(t + win) + 1))
            if not clip_secs & peak_secs:               # 피크(±5s)와 겹치지 않는 윈도우만 후보
                candidates.append(t)
            t += win                                    # 비중첩 스텝 (윈도우 크기만큼 이동)
        if not candidates:
            return []
        chosen = random.sample(candidates, min(count, len(candidates)))
        return [(s, s + win) for s in chosen]

    @staticmethod
    def _count_pos_per_video(pos_path: Path) -> dict:
        """포지티브 JSONL에서 영상당 샘플 수 집계 (metadata.video_id 기준, 네거티브 마커 제외)"""
 
        counts: dict = {}
        if not pos_path.is_file():
            return counts
        with open(pos_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata", {})
                vid = meta.get("video_id")
                if vid and meta.get("label") != "negative":     # 혹시 모를 혼합 파일 대비(네거티브 제외)
                    counts[vid] = counts.get(vid, 0) + 1
        return counts