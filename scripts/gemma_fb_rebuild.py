#!/usr/bin/env python3
"""
51일차: Gemma 피드백 충돌 미디어 재추출 (구간 다운로드 + 프레임/오디오 추출)

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_fb_rebuild.py
- 수정 이력: 1차 작성 (51일차)

[배경]
- 감사 결과(gemma_fb_audit.py): 103건 중 85건이 "source" 키 충돌로 미디어 덮어써짐
- 재추출 소스: DB youtube_url + 감사 매니페스트의 윈도우 start/end
- 검증된 부품 재사용: gemma_ytdlp.download_video_section(39일차, 구간만 다운로드)
  + extract_frames/extract_audio_segment(파이프라인과 동일 파라미터 → 분포 일치)

[경로 규칙 (파이프라인 수정 3회 규칙과 동일)]
  data/feedback_media_gemma/<project_id>_source/w<start:.0f>/{frames/, audio.wav}

[재개 지원]
- 이미 frames+audio.wav가 있는 윈도우는 스킵 → 중단 후 재실행 가능

[실행 전 필수]
  Node 22 활성화: export NVM_DIR="$HOME/.nvm" && source "$NVM_DIR/nvm.sh" && nvm use 22

[실행]
  python3 scripts/gemma_fb_rebuild.py
"""

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.core.gemma_config import gemma_settings
from app.services.frame_extractor import extract_frames
from app.services.gemma_audio_extractor import extract_audio_segment
from app.services.gemma_ytdlp import download_video_section

MANIFEST = Path("data/finetune/gemma_fb_audit.json")
MEDIA_ROOT = Path("data/feedback_media_gemma")
TMP_DIR = Path("data/finetune/_fb_rebuild_tmp")
DELAY_SEC = 4.0     # 51일차: 다운로드 간 딜레이 (레이트리밋 방어, 검증 권장값)


def _video_id_from_url(url: str) -> str | None:
    """51일차: watch?v=ID / youtu.be/ID 양쪽 형식에서 video id 추출"""

    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return m.group(1) if m else None


def _window_done(out_dir: Path) -> bool:
    """51일차: 재개 판정 - frames에 jpg 1장 이상 + audio.wav 존재"""

    frames_dir = out_dir / "frames"
    if not (out_dir / "audio.wav").is_file():
        return False
    return frames_dir.is_dir() and any(frames_dir.glob("*.jpg"))


async def _rebuild_window(video_id: str, w: dict, out_dir: Path) -> bool:
    """51일차: 1윈도우 재구축 - 구간 다운로드 -> 상대(0-base) 프레임/오디오 추출"""

    start = float(w["start_sec"])
    end = float(w["end_sec"])
    seg_len = end - start
    tmp_mp4 = TMP_DIR / f"{video_id}_{start:.0f}.mp4"

    try:
        # 39일차 검증 부품: 구간만 다운로드 (--force-keyframes-at-cuts -> 출력 0초 시작)
        await download_video_section(video_id, start, end, tmp_mp4)

        # 파이프라인(gemma_phase_inference._extract_window)과 동일 파라미터, 단
        # 구간 파일은 0초부터 시작하므로 상대 타임스탬프(0~seg_len) 사용
        frames = await extract_frames(
            video_path=tmp_mp4,
            interval_sec=1.0 / max(1, gemma_settings.GEMMA_FRAME_FPS),
            max_frames=gemma_settings.GEMMA_AUDIO_MAX_SEC,
            resolution=gemma_settings.GEMMA_FRAME_RESOLUTION,
            start_sec=0.0,
            end_sec=seg_len,
            save_dir=out_dir / "frames",
        )
        n_frames = sum(1 for f in frames if f.get("path"))
        audio = await extract_audio_segment(tmp_mp4, 0.0, seg_len, out_dir / "audio.wav")

        if n_frames == 0 or audio is None:
            logger.warning(f"  w{start:.0f}: 추출 불완전 (프레임 {n_frames}, 오디오 {audio})")
            return False
        logger.info(f"  w{start:.0f}: 프레임 {n_frames}장 + 오디오 완료")
        return True
    finally:
        tmp_mp4.unlink(missing_ok=True)     # 구간 파일은 추출 후 즉시 삭제 (디스크 절약)


async def main():
    if not MANIFEST.is_file():
        logger.error(f"매니페스트 없음: {MANIFEST} — gemma_fb_audit.py 먼저 실행")
        sys.exit(1)

    with open(MANIFEST, encoding="utf-8") as f:
        need: dict = json.load(f)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(len(p["windows"]) for p in need.values())
    logger.info(f"재추출 대상: {len(need)}개 영상 / {total}개 윈도우")

    done = skipped = failed = 0
    for proj_i, (project_id, proj) in enumerate(need.items(), 1):
        video_id = _video_id_from_url(proj["youtube_url"])
        if not video_id:
            logger.error(f"[{proj_i}/{len(need)}] URL에서 video id 추출 실패: {proj['youtube_url']}")
            failed += len(proj["windows"])
            continue
        logger.info(f"[{proj_i}/{len(need)}] {project_id[:8]} ({video_id}) "
                    f"윈도우 {len(proj['windows'])}개")

        # 51일차: 파이프라인 수정 3회 경로 규칙과 동일 (<project_id>_source)
        base_dir = MEDIA_ROOT / f"{project_id}_source"

        for w in proj["windows"]:
            out_dir = base_dir / f"w{float(w['start_sec']):.0f}"
            if _window_done(out_dir):
                skipped += 1
                continue
            try:
                ok = await _rebuild_window(video_id, w, out_dir)
            except Exception as exc:
                logger.warning(f"  w{float(w['start_sec']):.0f}: 실패 - {exc}")
                ok = False
            if ok:
                done += 1
            else:
                failed += 1
            await asyncio.sleep(DELAY_SEC)      # 레이트리밋 방어

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    logger.info("-" * 60)
    logger.info(f"완료 {done} / 스킵(기존) {skipped} / 실패 {failed} (총 {total})")
    if failed:
        logger.warning("실패 윈도우는 재실행 시 자동 재시도됩니다 (재개 지원)")


if __name__ == "__main__":
    asyncio.run(main())