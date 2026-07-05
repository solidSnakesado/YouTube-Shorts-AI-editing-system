# 48일차: gemma_e2e_collate.py — 신규 (전체 신규, 수정 0회)
# 레포 경로: yt_shorts_ai/scripts/gemma_e2e_collate.py
# 역할: 방안 1(round12) 데이터 파이프
#   - qwenfmt jsonl 1행 → 프로세서 입력(프레임+오디오+지시문) + y 실수 텐서
#   - gemma_collate.py(41일차)의 parse_sample/sample_frames/load_images/load_audio 재사용
#   - CE용 labels(-100 마스킹) 미사용 — 회귀이므로 y만 배치에 포함
#   - ★ 라벨 누설 방지: 타깃 텍스트를 입력 시퀀스에 절대 포함하지 않음
#     (user 턴만 렌더 + add_generation_prompt — 학습/추론 입력 형태 동일)

from __future__ import annotations

import json
import re
from typing import Any, Optional

from gemma_collate import load_audio, load_images, parse_sample, sample_frames

# 48일차: qwenfmt target JSON에서 hook_score 숫자 추출 (핸드오프 §5: _NUM_RE 방식)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def extract_score(target_text: str) -> float:
    """target 문자열에서 첫 숫자를 hook_score로 추출. 없으면 ValueError."""
    m = _NUM_RE.search(target_text)
    if m is None:
        raise ValueError(f"타깃에서 숫자 추출 실패: {target_text[:80]!r}")
    return float(m.group())


# ---------------------------------------------------------------------------
# jsonl 로더 (Dataset 대용 — 커스텀 루프에서 인덱싱)
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# collate_fn 팩토리 (회귀용)
# ---------------------------------------------------------------------------
def build_e2e_collate_fn(
    processor: Any,
    max_frames: int = 8,
    base_dir: Optional[str] = None,
    target_sr: int = 16000,
):
    """커스텀 학습 루프용 collate_fn 생성.

    Args:
        processor: Gemma4Processor (Colab 주입)
        max_frames: 샘플당 최대 프레임 수 (토큰 예산: 268토큰/프레임 선형,
                    8프레임 → max_length 3072 상당. 배치≥8 필요하므로 보수적 기본값)
        base_dir: jsonl 상대경로 기준 디렉토리 (/content 해제 위치)
    Returns:
        collate_fn: list[dict] → dict(프로세서 입력 + "y": FloatTensor[B])
    """

    def collate_fn(batch: list[dict]) -> dict:
        import torch

        texts: list[str] = []
        images_per_ex: list[list] = []
        audio_per_ex: list = []
        ys: list[float] = []

        for example in batch:
            parsed = parse_sample(example)
            frames = sample_frames(parsed["frame_paths"], max_frames)
            ys.append(extract_score(parsed["target"]))

            # 48일차: user 턴만 구성 — 타깃(정답 점수)은 입력에 미포함 (누설 방지)
            user_blocks: list[dict] = [{"type": "image"} for _ in frames]
            user_blocks.append({"type": "audio"})
            user_blocks.append({"type": "text", "text": parsed["instruction"]})
            messages = [{"role": "user", "content": user_blocks}]

            # add_generation_prompt=True: '<start_of_turn>model\n'까지 렌더
            #   → 학습·추론 입력 형태 완전 동일 (풀링 대상 시퀀스 정합)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            images_per_ex.append(load_images(frames, base_dir))
            wav, _sr = load_audio(parsed["audio_path"], target_sr, base_dir)
            audio_per_ex.append(wav)

        inputs = processor(
            text=texts, images=images_per_ex, audio=audio_per_ex,
            return_tensors="pt", padding=True,
        )
        # 48일차: CE labels 없음 — 회귀 타깃만
        inputs["y"] = torch.tensor(ys, dtype=torch.float32)
        return inputs

    return collate_fn


# ---------------------------------------------------------------------------
# 스모크 테스트 (Colab CPU — 프로세서 수용 + y 추출 확인, 0 CU)
# ---------------------------------------------------------------------------
def smoke_test(
    jsonl_path: str,
    processor: Any,
    max_frames: int = 8,
    base_dir: Optional[str] = None,
    n_samples: int = 2,
) -> dict:
    rows = load_jsonl(jsonl_path)[:n_samples]
    collate = build_e2e_collate_fn(processor, max_frames=max_frames, base_dir=base_dir)
    inputs = collate(rows)
    report = {
        "samples": len(rows),
        "seq_len": int(inputs["input_ids"].shape[1]),
        "keys": sorted(inputs.keys()),
        "y": [round(float(v), 3) for v in inputs["y"]],
        "input_ids_shape": tuple(inputs["input_ids"].shape),
    }
    print("=== E2E 회귀 collate 스모크 ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report