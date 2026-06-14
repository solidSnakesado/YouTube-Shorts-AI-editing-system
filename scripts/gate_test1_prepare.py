# 계층: 스크립트 (CLI 진입점)
# 역할: 게이트 진단 (1) 사람 동등 입력 테스트 - 모델 입력(5프레임 336px)과 100% 동일한
#       프레임만 보고 사람이 하이라이트/일반을 맹검(blind) 판정할 수 있는 지 검증 준비
# 의존: app.services.frame_extractor(extract_frames/_get_video_duration),
#       app.services.lora_utils(frames_to_pil),
#       app.services.phase2_inference(상수/전사 범위 추출), relabel_regression(히트맵 점수)
# 33일차 신규: 입력 충분성 게이트 - 사람도 5프레임으로 못 가르면 모델로 불가(표현력 한계 확정).
#   추출 경로를 추론과 동일하게 재사용해야 테스트가 유효함 (임의 추출 금지)

"""게이트 진단 (1) 준비 - 윈도우별 5프레임 몽타주 + 맹검 라벨 시트 + 숨김 정답 생성"""

import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path

# scripts/에서 'python scripts/...' 실행 시 프로젝트 루트를 path에 추가 (app 패키지 import용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from PIL import Image

from app.services.frame_extractor import extract_frames, _get_video_duration
from app.services.lora_utils import frames_to_pil
from app.services.phase2_inference import (
    FRAMES_PER_WINDOW,
    FRAME_RESOLUTION,
    WINDOW_SEC,
    _get_transcript_for_range,
)
from relabel_regression import clip_engagement, load_heatmaps

# 추론(phase2_inference line 93)과 동일한 프레임 추출 간격 - 변경 금지
INTERVAL_SEC = 2.0
# 맹검 테스트를 명확히 하기 위한 정답 버킷 경계 (중간 모호 구간은 표집 제외)
LOW_MAX = 0.40
HIGH_MIN = 0.60

def build_windows(duration: float) -> list[tuple[float, float]]:
    """영상을 WINDOW_SEC 단위 비중첩 윈도우로 분할 (정답 풀 생성용)"""
    
    windows = []
    t = 0.0
    while t + WINDOW_SEC <= duration:
        windows.append((t, t + WINDOW_SEC))
        t += WINDOW_SEC
    return windows

def make_montage(images: list, gap: int = 4) -> Image.Image | None:
    """5프레임 PIL 이미지를 가로로 이어붙인 단일 몽타주 생성 (사용자가 파악하기 위함)"""
    
    images = [im for im in images if im is not None]
    if not images:
        return None
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for im in images:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width + gap
    return canvas

async def _extract_window_images(video: Path, w_start: float, w_end: float) -> list:
    """추론과 동일 경로로 윈도우 5프레임 추출 -> PIL 리스트"""

    frames = await extract_frames(
        video,
        interval_sec=INTERVAL_SEC,
        start_sec=w_start,
        end_sec=w_end,
        max_frames=FRAMES_PER_WINDOW,
        resolution=FRAME_RESOLUTION,
    )
    return frames_to_pil(frames, max_count=FRAMES_PER_WINDOW)

def _sample_balanced(scored: list, num: int, no_dialogue_only: bool, seed: int) -> list:
    """저/고 참여도 균형 표집 (가능하면 무대사 구간 우선) - 모호 중간값 제외"""
    
    rng = random.Random(seed)
    if no_dialogue_only:
        pool = [w for w in scored if w["has_dialogue"] is False]
        if len(pool) < num:     # 무대사만으로 부족하면 전체로 보강
            logger.warning(f"무대사 윈도우 {len(pool)}개 < 요청 {num}개 -> 전체 풀로 보강")
            pool = scored
    else:
        pool = scored
    low = [w for w in pool if w["engagement"] <= LOW_MAX]
    high = [w for w in pool if w["engagement"] >= HIGH_MIN]
    rng.shuffle(low)
    rng.shuffle(high)
    half = num // 2
    picked = low[:half] + high[:num - half]
    rng.shuffle(picked)
    return picked

async def prepare(video, video_id, heatmap_path, transcript_path, num, no_dialogue_only, seed, out):
    """윈도우 점수화 -> 균형 표집 -> 몽타주/맹검시트/정답 생성"""
    
    heatmaps = load_heatmaps(Path(heatmap_path))
    if video_id not in heatmaps:
        raise SystemExit(f"video_id '{video_id}' 히트맵 없음 (총 {len(heatmaps)}개 영상)")
    segments = heatmaps[video_id]

    transcript_data = {}
    if transcript_path:
        transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))

    duration = await _get_video_duration(Path(video))

    # 윈도우별 정답 점수 + 대사 유무 태깅
    scored = []
    for w_start, w_end in build_windows(duration):
        eng = clip_engagement(segments, w_start, w_end)
        if eng is None:
            continue
        if transcript_data:
            transcript = _get_transcript_for_range(transcript_data, w_start, w_end)
            has_dialogue = bool(transcript.strip())
        else:
            has_dialogue = None     # 전사 미제공 -> 대사 유무 unknown
        scored.append({
            "start": w_start,
            "end": w_end,
            "engagement": round(eng, 2),
            "has_dialogue": has_dialogue,
        })

    picked = _sample_balanced(scored, num, no_dialogue_only, seed)
    if not picked:
        raise SystemExit("표집 가능한 윈도우 없음 (저/고 버킷 비어있음)")
    
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_rows = []
    answers = {}

    for idx, w in enumerate(picked):
        wid = f"w{idx:02d}"
        images = await _extract_window_images(Path(video), w["start"], w["end"])
        montage = make_montage(images)
        if montage is None:
            logger.warning(f"{wid}: 프레임 추출 실패, 건너뜀")
            continue
        montage.save(out_dir / f"{wid}.jpg", quality=90)
        time_range = f"{w['start']:.0f}-{w['end']:.0f}s"
        blind_rows.append({"window_id": wid, "time_range": time_range, "your_label(H/N)": ""})
        answers[wid] = {
            "engagement": w["engagement"],
            "true_label": "H" if w["engagement"] >= HIGH_MIN else "N",
            "has_dialogue": w["has_dialogue"],
            "time_range": time_range,
        }

    # 맹검 시트 (정답/대사 정보 미포함) - J가 your_label에 H/N 기입
    with open(out_dir / "label_blind.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "time_range", "your_label(H/N)"])
        writer.writeheader()
        writer.writerows(blind_rows)

    # 정답 (라벨링 끝나기 전 열람 금지)
    (out_dir / "answers.json").write_text(
        json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_high = sum(1 for a in answers.values() if a["true_label"] == "H")
    n_nodlg = sum(1 for a in answers.values() if a["has_dialogue"] is False)
    logger.info(
        f"준비 완료 | 윈도우 {len(answers)}개 (고:{n_high}/저:{len(answers) - n_high}) | 무대사:{n_nodlg}"
    )
    logger.info(f"출력: {out_dir} | 몽타주(w*.jpg) 보고 'label_blind.csv' 채운 뒤 채점 단계로")

def main() -> None:
    p = argparse.ArgumentParser(description="게이트 진단(1) 준비 - 사람 동등 입력 맹검 테스트 (33일차)")
    p.add_argument("--video", required=True, help="소스 영상 경로")
    p.add_argument("--video-id", required=True, help="heatmaps_merged.jsonl의 video_id")
    p.add_argument("--heatmap", default="data/heatmaps/heatmaps_merged.jsonl")
    p.add_argument("--transcript", default=None, help="Whisper 전사 JSON (무대사 태깅용, 선택)")
    p.add_argument("--num", type=int, default=24, help="표집 윈도우 수")
    p.add_argument("--no-dialogue-only", action="store_true", help="무대사 구간만 표집 (게임플레이 집중)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/gate_test1")
    args = p.parse_args()

    asyncio.run(prepare(
        args.video, args.video_id, args.heatmap, args.transcript,
        args.num, args.no_dialogue_only, args.seed, args.out,
    ))

if __name__ == "__main__":
    main()