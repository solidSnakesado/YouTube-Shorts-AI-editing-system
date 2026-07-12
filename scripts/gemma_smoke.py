# 55일차: gemma_smoke.py — 수정본 (수정 1회)
# 레포 경로: yt_shorts_ai/scripts/gemma_smoke.py
#   수정 1회(55일차): 더미 타깃을 빈 리스트 → build_highlight_output(0.0)으로 교체
#     (extract_score가 타깃에서 숫자를 요구 — 수정본 기준 L29~31, L66)
# 역할: .env의 GEMMA_INFER_ADAPTER_DIR 어댑터 스모크 검증 (라운드 전환 직후 필수 절차)
#   1) 어댑터 경로 출력 (round17/best 로드 확인)
#   2) data/feedback_media_gemma 기존 미디어 N개 윈도우 추론
#   3) 스코어 분포(mu/sd/min/max) 출력 — 상수 출력/비정상 값 즉시 탐지
# 실행 (레포 루트): uv run python scripts/gemma_smoke.py [--n 5]
# 의존: scripts/gemma_e2e_model.py, scripts/gemma_e2e_collate.py,
#   레포 루트 gemma_collate.py, app/services/gemma_sample.py

"""round17 어댑터 스모크 - 기존 피드백 미디어로 N윈도우 추론 후 분포 확인"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# 55일차: 레포 루트/scripts 경로 추가 (gemma_phase_inference와 동일 방식)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
for _p in (str(_REPO_ROOT), _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.core.gemma_config import gemma_settings                 # noqa: E402
from app.services.gemma_sample import (                          # noqa: E402
    HIGHLIGHT_INSTRUCTION, build_gemma_sample, build_highlight_output,
)

_MEDIA_DIR = _REPO_ROOT / "data" / "feedback_media_gemma"


def _find_windows(n: int) -> list[Path]:
    """55일차: frames/ + audio.wav 완비된 윈도우 디렉토리 N개 수집 (정렬 고정 → 재현성)"""

    found: list[Path] = []
    for src in sorted(_MEDIA_DIR.iterdir()):
        if not src.is_dir():
            continue
        for w in sorted(src.iterdir()):
            if not w.is_dir():
                continue
            frames = sorted((w / "frames").glob("frame_*.jpg"))
            audio = w / "audio.wav"
            if frames and audio.exists():
                found.append(w)
                if len(found) >= n:
                    return found
    return found


def _build_sample(win: Path) -> dict:
    """55일차: 윈도우 디렉토리 1개 → 추론용 messages 샘플 (경로는 레포 루트 상대)"""

    frames = sorted((win / "frames").glob("frame_*.jpg"))
    rel = lambda p: str(p.relative_to(_REPO_ROOT))                # noqa: E731
    return build_gemma_sample(
        frame_paths=[rel(f) for f in frames],
        audio_path=rel(win / "audio.wav"),
        instruction=HIGHLIGHT_INSTRUCTION,
        output_json=build_highlight_output(0.0),  # 더미 타깃 (collate가 숫자 요구, 입력 미포함)
        metadata={"window": win.name, "source": win.parent.name},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="round17 어댑터 스모크 검증")
    ap.add_argument("--n", type=int, default=5, help="추론할 윈도우 수 (기본 5)")
    args = ap.parse_args()

    adapter_dir = gemma_settings.GEMMA_INFER_ADAPTER_DIR
    print(f"[스모크] 어댑터 경로: {adapter_dir}")
    if "round17" not in adapter_dir:
        print("  ★ 경고: 경로에 round17이 없음 — .env 전환이 반영되지 않았을 수 있음")

    wins = _find_windows(args.n)
    if not wins:
        print(f"[중단] {_MEDIA_DIR} 에서 사용 가능한 윈도우 없음")
        sys.exit(1)
    print(f"[스모크] 윈도우 {len(wins)}개 수집 완료 — 스택 로드 시작 (최초 1~2분)")

    import torch                                                  # noqa: E402
    from gemma_e2e_collate import build_e2e_collate_fn            # scripts/
    from gemma_e2e_model import load_model_for_infer, load_norm_stats

    t0 = time.time()
    model, processor = load_model_for_infer(adapter_dir)          # PLE CPU 상주 포함
    model.head.to("cuda")
    collate = build_e2e_collate_fn(processor, max_frames=8)
    norm = load_norm_stats(adapter_dir)
    print(f"[스모크] 스택 로드 {time.time() - t0:.1f}s | norm mu={norm['mu']:.4f} sd={norm['sd']:.4f}")

    scores: list[float] = []
    for i, win in enumerate(wins, 1):
        t1 = time.time()
        batch = collate([_build_sample(win)])
        batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            p = model(batch).float().cpu().numpy()
        s = float(p[0] * norm["sd"] + norm["mu"])
        scores.append(s)
        print(f"  [{i}/{len(wins)}] {win.parent.name[:8]}…/{win.name} "
              f"score={s:.4f} ({time.time() - t1:.1f}s)")

    mu = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print("-" * 60)
    print(f"[결과] n={len(scores)} mu={mu:.4f} sd={sd:.4f} "
          f"min={min(scores):.4f} max={max(scores):.4f}")
    if sd < 1e-4 and len(scores) > 1:
        print("  ★ 판정: 상수 출력 의심 — 어댑터/헤드 로드 상태 점검 필요")
    else:
        print("  판정: 분포 정상 (상수 출력 없음) — 조기 분포 점검 게이트로 진행 가능")


if __name__ == "__main__":
    main()