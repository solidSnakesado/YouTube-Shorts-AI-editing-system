# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 히트맵 JSONL -> 피크/비피크 구간 프레임 추출 -> VLM 파인튜닝 데이터셋 생성
# 의존: config.py, frame_extractor.py, yt-dlp (외부 CLI)
# 21일차 신규: 영상 1개씩 다운로드 -> 프레임 추출 -> 삭제 (저장 공간을 효율적으로 사용하기 위함)
# 24일차 수정: generator 모드에서 다수 하이라이트를 하나의 샘플로 묶음

"""파인튜닝 데이터셋 빌더 - 히트맵 피크 기반 VLM 학습 데이터 생성"""

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.services.frame_extractor import extract_frames

class DatasetBuilder:
    """히트맵 피크 기반 VLM 파인튜닝 데이터셋 빌더"""

    def __init__(
        self, heatmap_path: Path, output_dir: Optional[Path] = None,
        min_peak_count: Optional[int] = None, frames_per_segment: Optional[int] = None,
        negative_ratio: Optional[float] = None, mode: str = "classifier",
    ):
        self.heatmap_path = Path(heatmap_path)
        self.mode = mode    # "classifier" (판별기) 또는 "generator" (생성기: 다수 JSON)
        self.output_dir = output_dir or settings.finetune_output_path
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.min_peak_count = min_peak_count or settings.FINETUNE_MIN_PEAK_COUNT
        self.frames_per_segment = frames_per_segment or settings.FINETUNE_FRAMES_PER_SEGMENT
        self.negative_ratio = negative_ratio if negative_ratio is not None else settings.FINETUNE_NEGATIVE_RATIO
        self._stats = {
            "total_videos": 0, "filtered_videos": 0, "processed": 0, 
            "skipped": 0, "positive_samples": 0, "negative_samples": 0,
        }

    async def build(self, output_filename: str = "dataset.jsonl") -> Path:
        """전체 파이프라인: JSONL 로드 -> 영상 선별 -> 프레임 추출 -> 데이터셋 저장"""

        output_path = self.output_dir / output_filename
        all_records = self._load_all()
        videos = [r for r in all_records if len(r.get("peak_segments", [])) >= self.min_peak_count]
        self._stats["total_videos"] = len(all_records)
        self._stats["filtered_videos"] = len(videos)
        logger.info(f"데이터셋 빌드 시작 | 전체: {len(all_records)}개 -> 선별: {len(videos)}개 (피크 {self.min_peak_count}개 이상)")
        for idx, video in enumerate(videos, 1):
            vid = video["video_id"]
            logger.info(f"[{idx}/{len(videos)}] 처리 중: {vid}")
            try:
                samples = await self._process_video(video)
                self._append_samples(output_path, samples)
                self._stats["processed"] += 1
                logger.info(f"[{idx}/{len(videos)}] 완료: {vid} | 샘플: {len(samples)}개")
            except Exception as e:
                logger.error(f"[{idx}/{len(videos)}] 실패: {vid} | {e}")
                self._stats["skipped"] += 1

        logger.info(
            f"데이터셋 빌드 완료 | 처리: {self._stats['processed']}개, 스킵: {self._stats['skipped']}개 | "
            f"포지티브: {self._stats['positive_samples']}개, 네거티브: {self._stats['negative_samples']}개"
        )
        return output_path

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _load_all(self) -> list[dict]:
        if not self.heatmap_path.is_file():
            raise FileNotFoundError(f"히트맵 JSONL 없음: {self.heatmap_path}")
        records = []
        with open(self.heatmap_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONL 파싱 오류 (라인 {line_num}): {e}")
        
        return records

    # --------------------------------------------------------------
    # 영상 단위 처리
    # --------------------------------------------------------------
    async def _process_video(self, video: dict) -> list[dict]:
        """다운로드 -> 피크/비피크 프레임 추출 -> 영상 삭제 -> 샘플 리스트 반환"""

        vid = video["video_id"]
        duration = video["duration_sec"]
        peaks = video.get("peak_segments", [])
        title = video.get("title", "")

        temp_dir = Path("temp") / "finetune"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{vid}.mp4"

        try:
            await self._download_video(vid, video_path)
            if not video_path.is_file():
                raise RuntimeError(f"다운로드 실패: {vid}")
            if self.mode == "generator":
                return await self._process_generator(video_path, vid, title, duration, peaks)
            return await self._process_classifier(video_path, vid, title, duration, peaks)
        finally:
            if video_path.is_file():
                video_path.unlink()
                logger.debug(f"임시 영상 삭제: {video_path}")

    async def _process_classifier(self, video_path: Path, vid: str, title:str, duration: float, peaks: list[dict]) -> list[dict]:
        """판별기 모드: 피크별 개별 샘플 생성"""

        samples = []
        for peak in peaks:
            frames = await self._extract_segment_frames(video_path, vid, peak["start_sec"], peak["end_sec"])
            if not frames:
                continue
            meta = self._seg_metadata(vid, title, duration, peak["start_sec"], peak["end_sec"])
            samples.append({
                "instruction": "이 게임 영상 프레임을 보고 시청자가 많이 다시 본 하이라이트 구간인지 판단하세요.",
                "images": frames, "metadata": meta, "output": "하이라이트"
            })
            self._stats["positive_samples"] += 1
        neg_count = int(len(peaks) * self.negative_ratio)
        for seg_s, seg_e in self._pick_negative_segments(peaks, duration, neg_count):
            frames = await self._extract_segment_frames(video_path, vid, seg_s, seg_e)
            if not frames:
                continue
            meta = self._seg_metadata(vid, title, duration, seg_s, seg_e)
            samples.append({
                "instruction": "이 게임 영상 프레임을 보고 시청자가 많이 다시 본 하이라이트 구간인지 판단하세요.",
                "images": frames, "metadata": meta, "output": "일반"
            })
            self._stats["negative_samples"] += 1
        return samples
    
    async def _process_generator(self, video_path: Path, vid: str, title:str, duration: float, peaks: list[dict]) -> list[dict]:
        """생성기 모드: 영상 전체 피크를 하나의 다수 하이라이트 샘플로 묶음"""

        norm_peaks = self._normalize_peaks(peaks)
        if not norm_peaks:
            return []
        all_frames: list[str] = []
        valid_peaks: list[dict] = []
        for peak in norm_peaks:
            frames = await self._extract_segment_frames(video_path, vid, peak["start_sec"], peak["end_sec"])
            if not frames:
                continue
            all_frames.extend(frames)
            valid_peaks.append(peak)
        if not valid_peaks:
            return []
        highlights = []
        for i, p in enumerate(valid_peaks):
            highlights.append({
                "start_sec": round(p["start_sec"], 1), "end_sec": round(p["end_sec"], 1),
                "hook_score": round(0.95 - i * 0.05, 2),
                "reason": "시청자가 많이 다시 본 구간",
                "title_suggestion": f"{title[:25]} #{i + 1}" if title else f"하이라이트 #{i + 1}",
                "tags": ["게임", "하이라이트"], "recommended_aspect_ratio": "16:9",
            })
        output = json.dumps({"highlights": highlights}, ensure_ascii=False)
        instruction = (
            "영상 프레임과 전사 텍스트를 분석하여 쇼츠 하이라이트 구간을 JSON으로 추출하세요. "
            "하이라이트가 없으면 빈 리스트를 반환하세요."
        )
        meta = {"video_id": vid, "video_title": title, "duration_sec": duration, "highlight_count": len(highlights)}
        self._stats["positive_samples"] += len(valid_peaks)
        return [{"instruction": instruction, "images": all_frames, "metadata": meta, "output": output}]

    # --------------------------------------------------------------
    # 프레임 추출
    # --------------------------------------------------------------

    async def _extract_segment_frames(self, video_path: Path, video_id: str, start: float, end: float) -> list[str]:
        """구간 프레임 추출 후 상대 경로 리스트 반환"""

        seg_duration = end - start
        if seg_duration <= 0:
            return []
        interval = seg_duration / max(self.frames_per_segment, 1)
        seg_dir = self.frames_dir / f"{video_id}_{int(start)}"
        results = await extract_frames(
            video_path=video_path, interval_sec=interval, max_frames=self.frames_per_segment,
            resolution=settings.FRAME_EXTRACT_RESOLUTION, start_sec=start, end_sec=end, save_dir=seg_dir,
        )
        
        frame_paths = []
        for r in results:
            abs_path = Path(r["path"])
            try:
                rel_path = abs_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = abs_path
            frame_paths.append(str(rel_path))
        return frame_paths

    # --------------------------------------------------------------
    # 헬퍼
    # --------------------------------------------------------------

    @staticmethod
    def _seg_metadata(vid: str, title: str, dur: float, start: float, end: float) -> dict:
        return {
            "video_id": vid, "video_title": title, "duration_sec": dur,
            "segment_start": start, "segment_end": end,
            "position_ratio": round(start / dur, 3) if dur > 0 else 0.0,
        }

    @staticmethod
    def _normalize_peaks(peaks: list[dict]) -> list[dict]:
        """구간 길이를 15~60초로 정규화"""

        normalized = []
        for p in peaks:
            start, end = p["start_sec"], p["end_sec"]
            dur = end - start
            if dur < 15.0:
                end = start + 30.0
            elif dur > 60.0:
                end = start + 60.0
            normalized.append({"start_sec": start, "end_sec": round(end, 1)})

        return normalized

    @staticmethod
    def _pick_negative_segments(peaks: list[dict], duration: float, count: int) -> list[tuple[float, float]]:
        """피크와 겹치지 않는 구간에서 네거티브 세그먼트 랜덤 선택"""

        if not peaks or count <= 0:
            return []
        avg_peak_len = sum(p["end_sec"] - p["start_sec"] for p in peaks) / len(peaks)
        seg_len = max(avg_peak_len, 10.0)

        peak_secs = set()
        for p in peaks:
            for s in range(int(p["start_sec"]), int(p["end_sec"]) + 1):
                peak_secs.add(s)

        candidates = []
        s = 0.0
        while s + seg_len <= duration:
            seg_secs = set(range(int(s), int(s + seg_len) + 1))
            if not seg_secs & peak_secs:
                candidates.append(s)
            s += seg_len

        if not candidates:
            return []
        chosen = random.sample(candidates, min(count, len(candidates)))
        return [(s, s + seg_len) for s in chosen]

    async def _download_video(self, video_id: str, output_path: Path) -> None:
        """yt-dlp로 최저 화질 다운로드 (프레임 추출용)"""

        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            "yt-dlp", "-f", "worst[ext=mp4]/worst",
            "-o", str(output_path), "--no-playlist",
            "--socket-timeout", str(settings.HEATMAP_REQUEST_TIMEOUT_SEC),
            "--no-warnings", url
        ]
        logger.info(f"영상 다운로드 시작: {video_id}")
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp 다운로드 실패 (rc={proc.returncode}): "
                f"{stderr.decode(errors='replace')[:300]}"
            )
        logger.info(f"영상 다운로드 완료: {video_id}")

    # --------------------------------------------------------------
    # JSONL 저장
    # --------------------------------------------------------------
    @staticmethod
    def _append_samples(output_path: Path, samples: list[dict]) -> None:
        if not samples:
            return
        with open(output_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")