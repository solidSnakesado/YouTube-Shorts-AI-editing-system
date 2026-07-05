# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 히트맵 JSONL ->피크별 구간 다운로드 -> [1fps 프레임 + 오디오] -> Gemma messages 데이터셋 생성
#       Gemma 4 E4B 오디오 피벗 데이터 재구축 - 모듈 C-2b (메인 빌더)
# 의존: gemma_ytdlp.download_video_section(C-1), gemma_audio_extractor(모듈 A), frame_extractor(재사용),
#       gemma_sample(C-2a), dataset_utils.load_processed_ids(재사용), gemma_config
# 39일차 신규: 풀영상 다운로드 병목(롱플레이 수 시간) 해소 -> 피크별 구간만 다운로드(--download-sections).
#   - 구간 파일은 0초 시작 -> 추출은 상대(0-base) 시각, 메타/출력 레이블은 절대 시각(원본 영상 위치) 유지
#   - 피크별 다운로드 실패를 peak_dl_fails로 가시화 (조용한 누락 방지, 피크별 실패 격리)
# 45일차 수정(1회): 영상 간 delay 추가(403 rate limit 방어). 빠른 연속 요청 -> YouTube IP 일시차단(403)
#   -> 90초 대기 후 동일 영상 성공 확인됨(=rate limit). 다운로드한 영상 뒤에만 sleep(스킵은 즉시 진행).
#   변경: __init__ delay 인자 + 루프 sleep. run_gemma_build에 --delay 인자. 변경 라인은 전달 메시지 참조.

"""Gemma 오디오 피벗 데이터셋 빌더 - 히트맵 -> [1fps 프레임 + 30s 오디오] -> messages JSONL"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.gemma_config import gemma_settings
from app.services.frame_extractor import extract_frames
from app.services.dataset_utils import load_processed_ids
from app.services.gemma_ytdlp import download_video_section
from app.services.gemma_audio_extractor import extract_audio_segment
from app.services.gemma_sample import build_highlight_output, build_gemma_sample, HIGHLIGHT_INSTRUCTION

class GemmaDatasetBuilder:
    """히트맵 피크 기반 Gemma 오디오 데이터셋 빌더 (Qwen DatasetBuilder와 병렬, 격리)"""

    def __init__(
        self,
        heatmap_path: Path,
        output_dir: Optional[Path] = None,
        min_peak_count: int = 2,
        max_videos: Optional[int] = None,
        delay: float = 0.0,
    ):
        self.heatmap_path = Path(heatmap_path)
        # 출력 루트 기본값: datasets/gemma_audio (config 추가 없이 파라미터로 제어)
        self.output_dir = Path(output_dir) if output_dir else Path("datasets/gemma_audio")
        self.frames_dir = self.output_dir / "frames"
        self.audio_dir = self.output_dir / "audio"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.min_peak_count = min_peak_count
        self.max_videos = max_videos
        self.delay = delay                              # 45일차: 영상 간 대기(초), 403 rate limit 방어
        self._stats = {
            "total_videos": 0, "filtered_videos": 0, "processed": 0,
            "skipped": 0, "samples": 0, "no_audio_skips": 0, "peak_dl_fails": 0,
        }

    async def build(self, output_filename: str = "dataset.jsonl") -> Path:
        """전체 파이프라인: 히트맵 로드 -> 영상 선별 -> 다운로드 -> 클립 -> messages 저장 (재개 가능)"""

        output_path = self.output_dir / output_filename
        processed_ids = load_processed_ids(output_path)     # metadata.video_id 기준 재개
        if processed_ids:
            logger.info(f"이어서 진행 | 기처리 {len(processed_ids)}개 스킵")
        all_records = self._load_all()
        videos = [r for r in all_records if len(r.get("peak_segments", [])) >= self.min_peak_count]
        if self.max_videos:
            videos = videos[: self.max_videos]
        self._stats["total_videos"] = len(all_records)
        self._stats["filtered_videos"] = len(videos)
        logger.info(
            f"Gemma 데이터셋 빌드 시작 | 전체 {len(all_records)} -> 선별 {len(videos)} (피크 {self.min_peak_count}+)"
        )
        for idx, video in enumerate(videos, 1):
            vid = video["video_id"]
            if vid in processed_ids:
                self._stats["processed"] += 1
                continue
            logger.info(f"[{idx}/{len(videos)}] 처리 중: {vid}")
            try:
                samples = await self._process_video(video)
                self._append_samples(output_path, samples)
                self._stats["processed"] += 1
                logger.info(f"[{idx}/{len(videos)}] 완료: {vid} | 샘플 {len(samples)}개")
            except Exception as e:
                logger.error(f"[{idx}/{len(videos)}] 실패: {vid} | {e}")
                self._stats["skipped"] += 1
            if self.delay > 0:                          # 45일차: 실제 요청 후에만 대기(403 방어)
                await asyncio.sleep(self.delay)
        logger.info(
            f"Gemma 데이터셋 빌드 완료 | 처리 {self._stats['processed']}, 스킵 {self._stats['skipped']} | "
            f"샘플 {self._stats['samples']}개, 오디오없음 스킵 {self._stats['no_audio_skips']}개, "
            f"피크 다운로드 실패 {self._stats['peak_dl_fails']}개"
        )
        return output_path

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _load_all(self) -> list[dict]:
        """히트맵 JSONL 로드 (Qwen 빌더와 동일 포맷). video_id 중복 시 peak 최다 1건만 유지."""

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
        # 39일차: video_id 중복 제거. 히트맵 재병합 등으로 동일 영상이 여러 줄이면
        #   네거티브가 video_id 필터로 중복 항목까지 처리 -> 영상 2배 버그(초과 매칭).
        #   peak 최다 항목 유지(동수는 첫 항목) -> 저peak 항목 잘못 채택 방지. 양쪽 빌더 공통 적용.
        best: dict = {}
        no_vid: list = []
        for r in records:
            vid = r.get("video_id")
            if vid is None:
                no_vid.append(r)
                continue
            if vid not in best or len(r.get("peak_segments", [])) > len(best[vid].get("peak_segments", [])):
                best[vid] = r
        deduped = list(best.values()) + no_vid
        if len(deduped) < len(records):
            logger.info(f"히트맵 video_id 중복 제거: {len(records)} -> {len(deduped)}건")
        return deduped

    async def _process_video(self, video: dict) -> list[dict]:
        """피크별로 구간 다운로드 -> [프레임+오디오] 추출 -> 구간 파일 삭제. 피크별 실패 격리"""

        vid = video["video_id"]
        duration = video.get("duration_sec", 0.0)
        title = video.get("title", "")
        peaks = self._normalize_peaks(video.get("peak_segments", []))
        if not peaks:
            return []
        temp_dir = Path("temp") / "gemma"
        temp_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict] = []
        for pi, peak in enumerate(peaks, 1):
            section_path = temp_dir / f"{vid}_{int(peak['start_sec'])}.mp4"
            try:
                # C-1: 피크 구간만 다운로드 (절대 시각). 출력 파일은 0초부터 시작.
                await download_video_section(vid, peak["start_sec"], peak["end_sec"], section_path)        
                if not section_path.is_file():
                    raise RuntimeError("구간 다운로드 산출물 없음")
                samples.extend(
                    await self._build_peak_clips(section_path, vid, title, duration, peak)
                )
            except Exception as e:
                self._stats["peak_dl_fails"] += 1           # 39일차: 피크별 실패 가시화
                logger.warning(
                    f"피크 구간 실패: {vid} [{pi}/{len(peaks)}] {peak['start_sec']:.0f}s | {e}"
                )
            finally:
                if section_path.is_file():
                    section_path.unlink()
        return samples

    async def _build_peak_clips(
        self, section_path: Path, vid: str, title: str, duration: float, peak: dict,
    ) -> list[dict]:
        """0초 시작 구간 파일에서 30s 클립 추출. 추출=상대(0-base) 시각, 메타/출력=절대 시각."""

        instruction = HIGHLIGHT_INSTRUCTION             # 39일차: pos/neg 공유 상수 (양쪽 동일, 드리프트 방지)
        win = gemma_settings.GEMMA_AUDIO_MAX_SEC        # 30s 정렬 윈도우
        peak_start, peak_end = peak["start_sec"], peak["end_sec"]
        samples: list[dict] = []
        clip_start = peak_start
        while clip_start < peak_end:
            clip_end = min(clip_start + win, peak_end)
            if clip_end - clip_start < win * 0.5:
                break                               # 잔여 클립이 윈도우 절반 미만이면 스킵
            rel_start = clip_start - peak_start     # 구간 파일 기준 0-base (seek용)
            rel_end = clip_end - peak_start
            # 오디오 먼저: 트랙 없으면 클립 자체 스킵 (오디오 모델이므로 오디오 필수)
            audio_rel = await self._extract_audio(section_path, vid, clip_start, rel_start, rel_end)
            if audio_rel is None:
                self._stats["no_audio_skips"] += 1
                clip_start = clip_end
                continue
            frames = await self._extract_frames(section_path, vid, clip_start, rel_start, rel_end)
            if not frames:
                clip_start = clip_end
                continue
            # 39일차: 레이블=hook_score만. 메타는 절대 시각(추적용, 모델 입력 아님)
            output = build_highlight_output(peak.get("avg_value", 0.5))
            meta = {
                "video_id": vid, "video_title": title, "duration_sec": duration,
                "clip_start": round(clip_start, 1), "clip_end": round(clip_end, 1)
            }
            samples.append(build_gemma_sample(frames, audio_rel, instruction, output, meta))
            self._stats["samples"] += 1
            clip_start = clip_end
        return samples

    async def _extract_frames(
        self, video_path: Path, vid: str, abs_start: float, rel_start: float, rel_end: float,
    ) -> list[str]:
        """1fps 프레임 추출 파일명=절대 시각, seek=상대(0-base) 시각 -> 상대 경로 리스트"""

        seg_dir = self.frames_dir / f"{vid}_{int(abs_start)}"
        results = await extract_frames(
            video_path=video_path,
            interval_sec=1.0 / gemma_settings.GEMMA_FRAME_FPS,      # 1fps -> interval 1.0
            max_frames=gemma_settings.GEMMA_AUDIO_MAX_SEC,          # 30s 윈도우 = 최대 30프레임
            resolution=gemma_settings.GEMMA_FRAME_RESOLUTION,
            start_sec=rel_start, end_sec=rel_end, save_dir=seg_dir,
        )
        return [self._rel(r["path"]) for r in results]

    async def _extract_audio(
        self, video_path: Path, vid: str, abs_start: float, rel_start: float, rel_end: float,
    ) -> Optional[str]:
        """30s 오디오 세그먼트 추출(모듈 A). 파일명=절대 시각, seek=상대 -> 상대 경로 (무음 시 None)"""

        out = self.audio_dir / f"{vid}_{int(abs_start)}.wav"        # 파일명은 절대 시각 기준
        saved = await extract_audio_segment(video_path, rel_start, rel_end, out)
        return self._rel(saved) if saved else None

    @staticmethod
    def _rel(path) -> str:
        """절대 경로 -> cwd 기준 상대 경로 문자열 (Colab 업로드 호환)"""

        p = Path(path)
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return str(p)

    @staticmethod
    def _normalize_peaks(peaks: list[dict]) -> list[dict]:
        """피크 길이 정규화 (짧으면 30s 확장 / 60s 초과는 60s 캡) - Qwen 빌더와 동일 취지"""

        normalized = []
        for p in peaks:
            start, end = p["start_sec"], p["end_sec"]
            dur = end - start
            if dur < 15.0:
                end = start + 30.0
            elif dur > 60.0:
                end = start + 60.0
            normalized.append({
                "start_sec": start, "end_sec": round(end, 1),
                "avg_value": p.get("avg_value", 0.5),
            })
        return normalized

    @staticmethod
    def _append_samples(output_path: Path, samples: list[dict]) -> None:
        if not samples:
            return
        with open(output_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")