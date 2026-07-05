#!/usr/bin/env python3
"""
51일차: Gemma 피드백 -> 학습 JSONL 빌더 (B-2 상대 너지 ±0.15)

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_fb_build.py
- 수정 이력: 1차 작성 (51일차) -> 수정 1회(51일차): yt_id 메타 추가 (수정본 기준 L31, L61~L68, L74~L78, L199~L200) -> 수정 2회(51일차): 경로 상대화 _rel 추가·적용 (수정본 기준 L115~L124, L135~L136)

[핵심 규칙 — 34일차 B-2 상대 너지 (round4~11 재붕괴 교훈으로 필수)]
- OK -> 원래 예측(hook_score) + 0.15 / NO(selection) -> 원래 예측 − 0.15, [0,1] 클램프
- 고정 0.9/0.1 금지: 윈도우별 원점수가 달라 타깃 연속 분포 유지 -> 이진 붕괴 방지
- NO 사유 boundary/editing은 기본 제외 (구간 선택은 옳았음 -> 선택 신호 오염 방지,
  Qwen build_feedback_dataset.py와 동일 규칙. --include-nonselection-no로 포함 가능)

[미디어 해석 순서]
1. train_sample_json의 video_id == "source"(충돌) -> 재구축 경로 사용
   (data/feedback_media_gemma/<project_id>_source/w<start:.0f>/)
2. 그 외(정상 18건) -> 저장된 경로 실존 검증 후 그대로 사용, 실패 시 재구축 경로 폴백

[출력]
- datasets/gemma_audio/dataset_feedback_r1.jsonl (v2 학습 스키마와 동일: messages+metadata)
- metadata.video_id = "<project_id>_source" -> package_dataset의 video-level split 그대로 호환

[실행]
  python3 scripts/gemma_fb_build.py
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.services.gemma_sample import (
    HIGHLIGHT_INSTRUCTION, build_gemma_sample, build_highlight_output,
)

DB_PATH = Path("data/shorts_ai.db")
MEDIA_ROOT = Path("data/feedback_media_gemma")
DEFAULT_OUT = "datasets/gemma_audio/dataset_feedback_r1.jsonl"

NUDGE_DELTA = 0.15                              # 34일차 B-2 확정값
_NEUTRAL_BASE = 0.5                             # hook_score 결측 시 중립 기준
_NONSELECTION_REASONS = {"boundary", "editing"}
_MIN_SAMPLES, _MIN_VIDEOS, _MIN_CLASS_RATIO = 100, 3, 0.30     # Qwen 빌더와 동일 임계


def _fetch_rows() -> list[dict]:
    """51일차: round12 피드백 행 로드 (project_id 포함)"""

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        # 51일차 수정 1회: projects 조인 -> youtube_url (yt_id 메타용, split 누수 검출 근거)
        rows = conn.execute(
            "SELECT s.id, s.project_id, s.feedback, s.feedback_reason, s.is_exploration, "
            "s.hook_score, s.model_version, s.train_sample_json, p.youtube_url "
            "FROM shorts s JOIN projects p ON p.id = s.project_id "
            "WHERE s.model_version LIKE '%round12%' AND s.feedback IS NOT NULL "
            "AND s.train_sample_json IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _yt_id(url: str) -> str:
    """51일차 수정 1회: watch?v=ID / youtu.be/ID -> YouTube 11자 ID (실패 시 빈 문자열)"""

    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else ""


def _label_score(label: str, reason, base, include_nonsel: bool) -> float | None:
    """51일차: B-2 상대 너지 - OK -> +Δ / NO(selection) -> −Δ, 비선택NO는 None(제외)"""

    if label == "NO" and (not include_nonsel) and reason in _NONSELECTION_REASONS:
        return None
    b = base if isinstance(base, (int, float)) else _NEUTRAL_BASE
    delta = NUDGE_DELTA if label == "OK" else -NUDGE_DELTA
    return round(max(0.0, min(1.0, b + delta)), 4)


def _old_media(sample: dict) -> tuple[list[str], str | None, str]:
    """51일차: 기존 샘플에서 (프레임들, 오디오, video_id) 추출"""

    frames, audio = [], None
    for msg in sample.get("messages", []):
        if msg.get("role") != "user":
            continue
        for b in msg.get("content", []):
            if b.get("type") == "image":
                frames.append(b.get("image", ""))
            elif b.get("type") == "audio":
                audio = b.get("audio")
    return frames, audio, str(sample.get("metadata", {}).get("video_id", ""))


def _rebuilt_media(project_id: str, start: float) -> tuple[list[str], str | None]:
    """51일차: 재구축 경로에서 (프레임들, 오디오) 로드 - frame_%04d 제로패딩이라 사전순 안전"""

    d = MEDIA_ROOT / f"{project_id}_source" / f"w{start:.0f}"
    frames = sorted(str(p) for p in (d / "frames").glob("frame_*.jpg"))
    audio = d / "audio.wav"
    return frames, (str(audio) if audio.is_file() else None)


def _rel(p: str) -> str:
    """51일차 수정 2회: 절대 경로 -> 레포 루트(cwd) 기준 상대 경로 (Colab base_dir 해석용)"""

    q = Path(p)
    if not q.is_absolute():
        return p
    try:
        return str(q.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return p        # 레포 밖 경로면 원본 유지 (검증 단계에서 드러남)


def _resolve_media(row: dict, sample: dict, start: float) -> tuple[list[str], str | None]:
    """51일차: 미디어 해석 - 충돌이면 재구축 경로, 정상이면 기존 경로(검증) -> 폴백"""

    frames, audio, video_id = _old_media(sample)
    if video_id != "source":
        ok = frames and all(Path(p).is_file() for p in frames) \
            and audio and Path(audio).is_file()
        if ok:
            # 51일차 수정 2회: 파이프라인 저장분(절대 경로)을 상대화 (Colab 이식성)
            return [_rel(p) for p in frames], _rel(audio)
    return _rebuilt_media(row["project_id"], start)


def _report(records: list, skipped: dict) -> None:
    """51일차: 클래스 균형/영상 다양성/임계값 점검 (Qwen 빌더와 동일 항목)"""

    labels = Counter(r["metadata"]["feedback"] for r in records)
    videos = {r["metadata"]["video_id"] for r in records}
    scores = [json.loads(r["messages"][1]["content"][0]["text"])
              ["highlights"][0]["hook_score"] for r in records]
    n = len(records)

    logger.info("-" * 60)
    logger.info(f"변환 완료: {n}개 | OK:{labels.get('OK', 0)} NO:{labels.get('NO', 0)} "
                f"| 영상:{len(videos)}개")
    if scores:
        logger.info(f"타겟 분포: min {min(scores):.3f} / mean {sum(scores)/n:.3f} "
                    f"/ max {max(scores):.3f}")
    logger.info(f"제외: 비선택NO {skipped['nonsel']} | 미디어 {skipped['media']} "
                f"| 파싱 {skipped['parse']}")

    if n < _MIN_SAMPLES:
        logger.warning(f"샘플 {n} < {_MIN_SAMPLES} - 재학습엔 소량 (2차 학습은 v2와 병합 전제)")
    if len(videos) < _MIN_VIDEOS:
        logger.warning(f"영상 {len(videos)} < {_MIN_VIDEOS} - 다양성 부족")
    if n > 0:
        minor = min(labels.get("OK", 0), labels.get("NO", 0)) / n
        if minor < _MIN_CLASS_RATIO:
            logger.warning(f"소수 클래스 {minor:.0%} < {_MIN_CLASS_RATIO:.0%} "
                           f"- 업샘플링/손실가중 검토 필요")


def build(output: Path, include_nonsel: bool) -> None:
    rows = _fetch_rows()
    logger.info(f"피드백 행 로드: {len(rows)}건")

    records, skipped = [], {"nonsel": 0, "media": 0, "parse": 0}
    for row in rows:
        label = (row["feedback"] or "").upper()
        score = _label_score(label, row["feedback_reason"], row["hook_score"], include_nonsel)
        if score is None:
            skipped["nonsel"] += 1
            continue
        try:
            old = json.loads(row["train_sample_json"])
            meta_old = old.get("metadata", {})
            start = float(meta_old["start_sec"])
            end = float(meta_old["end_sec"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(f"train_sample_json 파싱 실패: {row['id']}")
            skipped["parse"] += 1
            continue

        frames, audio = _resolve_media(row, old, start)
        if not frames or audio is None:
            logger.warning(f"미디어 미해석 (재구축 누락?): {row['id']} w{start:.0f}")
            skipped["media"] += 1
            continue

        # 51일차: 학습 스키마 그대로 재조립 (타겟 = 너지 반영 hook_score)
        record = build_gemma_sample(
            frame_paths=frames,
            audio_path=audio,
            instruction=HIGHLIGHT_INSTRUCTION,
            output_json=build_highlight_output(score),
            metadata={
                "video_id": f"{row['project_id']}_source",      # video-level split 키
                "start_sec": start, "end_sec": end,
                "source": "feedback",                           # 추적용 (학습 미사용)
                "feedback": label,
                "feedback_reason": row["feedback_reason"],
                "is_exploration": bool(row["is_exploration"]),
                "model_version": row["model_version"],
                "base_score": row["hook_score"],                # 너지 전 원점수 (검증용)
                "shorts_id": row["id"],
                # 51일차 수정 1회: YouTube 11자 ID (v2 학습 영상과의 겹침=누수 검출용)
                "yt_id": _yt_id(row["youtube_url"]),
            },
        )
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _report(records, skipped)
    logger.info(f"출력: {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma 피드백 -> 학습 JSONL (51일차)")
    p.add_argument("--output", default=DEFAULT_OUT, help="출력 JSONL 경로")
    p.add_argument("--include-nonselection-no", action="store_true",
                   help="boundary/editing NO도 −Δ 포함 (기본: 제외)")
    args = p.parse_args()
    if not DB_PATH.exists():
        raise SystemExit(f"DB 없음: {DB_PATH}")
    build(Path(args.output), args.include_nonselection_no)


if __name__ == "__main__":
    main()