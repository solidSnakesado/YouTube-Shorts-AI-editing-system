#!/usr/bin/env python3
"""
51일차: Gemma round12 피드백 샘플 미디어 상태 감사

[스크립트 정보]
- 구분: 신규 (전체 신규)
- 레포 경로: scripts/gemma_fb_audit.py
- 수정 이력: 1차 작성 (51일차)

[배경]
- 50일차 핸드오프 §6: gemma_phase_inference 수정 3회 이전 라벨은 보존 경로 키가
  "source"로 충돌하여 미디어(frames/audio)가 서로 덮어써짐
- 재학습 빌더 설계 전에 103건(usable 101 + 경계NO 2)의 상태를 분류해야 함

[분류 기준]
- 정상   : video_id가 "source"가 아니고, 프레임 전량 + 오디오 파일 실존
- 충돌   : video_id == "source" (파일이 있어도 다른 영상 것일 수 있어 신뢰 불가)
- 유실   : video_id는 정상이나 프레임/오디오 파일 일부 부재

[출력]
- 콘솔 리포트 + data/finetune/gemma_fb_audit.json (재추출 매니페스트:
  프로젝트별 youtube_url + 재추출 필요 윈도우 목록)

[실행]
  python3 scripts/gemma_fb_audit.py
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("data/shorts_ai.db")
OUT_PATH = Path("data/finetune/gemma_fb_audit.json")


def _extract_media(sample: dict) -> tuple[list[str], str | None]:
    """51일차: Gemma messages 샘플에서 (프레임 경로들, 오디오 경로) 추출"""

    frames: list[str] = []
    audio: str | None = None
    for msg in sample.get("messages", []):
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "image":
                frames.append(block.get("image", ""))
            elif block.get("type") == "audio":
                audio = block.get("audio")
    return frames, audio


def _classify(video_id: str, frames: list[str], audio: str | None) -> str:
    """51일차: 샘플 1건 상태 분류 (정상/충돌/유실)"""

    if video_id == "source":
        return "충돌"                       # 파일 존재 여부와 무관하게 신뢰 불가
    frames_ok = bool(frames) and all(Path(p).is_file() for p in frames)
    audio_ok = audio is not None and Path(audio).is_file()
    return "정상" if (frames_ok and audio_ok) else "유실"


def main():
    if not DB_PATH.exists():
        print(f"[오류] DB 없음: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 51일차: round12 라벨 완료 행 전량 (usable + 경계NO 포함 — 포함 여부는 빌더에서 결정)
    cur.execute(
        """
        SELECT s.id, s.project_id, s.feedback, s.feedback_reason, s.is_exploration,
               s.hook_score, s.train_sample_json, p.youtube_url
        FROM shorts s JOIN projects p ON p.id = s.project_id
        WHERE s.model_version LIKE '%round12%' AND s.feedback IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    counts = defaultdict(int)
    reasons = defaultdict(int)
    # 51일차: 재추출 매니페스트 - 프로젝트별 {url, windows[]}
    need: dict[str, dict] = {}
    parse_fail = 0

    for r in rows:
        try:
            sample = json.loads(r["train_sample_json"])
        except (json.JSONDecodeError, TypeError):
            parse_fail += 1
            continue
        meta = sample.get("metadata", {})
        video_id = str(meta.get("video_id", ""))
        w_start = meta.get("start_sec")
        w_end = meta.get("end_sec")
        frames, audio = _extract_media(sample)

        state = _classify(video_id, frames, audio)
        counts[state] += 1
        reasons[f"{r['feedback']}/{r['feedback_reason'] or '-'}"] += 1

        if state != "정상":
            proj = need.setdefault(
                r["project_id"],
                {"youtube_url": r["youtube_url"], "windows": []},
            )
            proj["windows"].append(
                {
                    "shorts_id": r["id"],
                    "start_sec": w_start,
                    "end_sec": w_end,
                    "state": state,
                }
            )

    # ---- 콘솔 리포트 ----
    print("=" * 60)
    print(f"라벨 완료 round12 행: {len(rows)}건 (파싱 실패 {parse_fail})")
    print(f"상태 분류: 정상 {counts['정상']} / 충돌 {counts['충돌']} / 유실 {counts['유실']}")
    print("-" * 60)
    print("라벨 분포 (feedback/사유):")
    for k, v in sorted(reasons.items()):
        print(f"  {k:20s}: {v}")
    print("-" * 60)
    total_win = sum(len(p["windows"]) for p in need.values())
    print(f"재추출 필요: {len(need)}개 영상 / {total_win}개 윈도우")
    for pid, p in need.items():
        print(f"  {pid[:8]}  윈도우 {len(p['windows'])}개  {p['youtube_url']}")

    # ---- 매니페스트 저장 (다음 단계 재추출 스크립트 입력) ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(need, f, ensure_ascii=False, indent=2)
    print(f"\n매니페스트 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()