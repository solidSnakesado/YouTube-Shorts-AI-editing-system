"""42일차: Gemma 4 E4B 멀티모달 추론 모듈 (baseline 어댑터 -> hook_score).

학습(gemma_collate.py)과 동일 전처리를 재사용해 학습-추론 일치를 보장한다.
추론은 학습과 달리 user 메시지만 넣고 add_generation_prompt=True로 응답을 생성한다.
타임스탬프는 모델 출력이 아니라 클립 윈도우에서 재구성한다(호출부 책임).

단독 실행 시(__main__) all.jsonl 첫 행으로 1샘플 추론을 수행하여
로컬 12GB VRAM에 멀티모달 추론이 들어가는지(관문 a)를 실측한다.

배치 경로: 저장소 루트 ~/project/yt_shorts_ai/gemma_inference.py (독립 모듈, 신규).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# 42일차: 학습 collate의 입력 구성 함수 재사용 (학습-추론 일치의 핵심)
from gemma_collate import parse_sample, sample_frames, load_images, load_audio

# 42일차: 기본 경로/하이퍼파라미터 (학습과 반드시 일치)
ADAPTER_DIR = "models/lora/gemma4/baseline_r1"  # 학습 어댑터(base+LoRA 직접 로드)
MAX_FRAMES = 8        # 학습 max_frames=8과 동일 (불일치 시 결과 엉킴)
MAX_SEQ_LENGTH = 3072   # 학습 입력 ~3000토큰 수용 (8프레임+오디오+프롬프트)
MAX_NEW_TOKENS = 64   # 출력은 짧은 JSON: {"highlights": [{"hook_score": 0.xxxx}]}


def load_gemma(adapter_dir: str = ADAPTER_DIR):
    """LoRA 어댑터 직접 로드 -> (model, processor).

    42일차: 검증된 기존 로딩 방식(app/services/lora_utils.py)을 따른다.
    - FastVisionModel(멀티모달용, 4bit가 올바로 적용됨) 사용
    - model_name에 어댑터 경로를 직접 지정 -> unsloth가 base+LoRA를 올바른
      구조로 함께 로드(별도 load_adapter는 레이어 불일치/MISSING 유발하므로 미사용)
    - device_map 미지정 -> unsloth 기본 배치(과보수적 오프로드/bf16 승격 방지)
    이전 방식(FastModel + load_adapter + device_map)은 어댑터 MISSING +
    4bit 미적용으로 VRAM 폭증(21.97GB)을 유발했었다.

    프로세서는 어댑터 폴더에서 로드(학습 때 함께 저장됨).
    """
    from unsloth import FastVisionModel
    from transformers import AutoProcessor

    # 42일차: 어댑터 경로를 model_name에 직접 -> base+LoRA 동시 로드(4bit)
    model, _tok = FastVisionModel.from_pretrained(
        model_name=adapter_dir,                 # 학습 어댑터(baseline_r1) 직접 지정
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    FastVisionModel.for_inference(model)         # 추론 모드(2x)

    processor = AutoProcessor.from_pretrained(adapter_dir)
    return model, processor


def build_inference_inputs(processor: Any, sample: dict, max_frames: int = MAX_FRAMES,
                           base_dir: Optional[str] = None):
    """jsonl 1행 -> 추론용 프로세서 입력. user만 + 생성 프롬프트.

    학습 collate와 동일하게 프레임을 max_frames로 샘플링하고 image 플레이스홀더
    개수를 실제 전달 이미지 수와 일치시킨다. 차이점은 assistant 블록을 넣지 않고
    add_generation_prompt=True로 모델이 응답을 생성하게 하는 것.
    """
    parsed = parse_sample(sample)
    frames = sample_frames(parsed["frame_paths"], max_frames)

    user_blocks: list[dict] = [{"type": "image"} for _ in frames]
    user_blocks.append({"type": "audio"})
    user_blocks.append({"type": "text", "text": parsed["instruction"]})
    messages = [{"role": "user", "content": user_blocks}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    images = load_images(frames, base_dir)
    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)

    inputs = processor(
        text=[text], images=[images], audio=[wav],
        return_tensors="pt", padding=True,
    )
    return inputs, parsed


def parse_hook_score(generated_text: str) -> float:
    """생성 텍스트 -> hook_score.

    {"highlights": []}                          -> 0.0  (비하이라이트)
    {"highlights": [{"hook_score": 0.9551}]}    -> 0.9551
    파싱 실패                                    -> -1.0 (오류 신호)
    """
    m = re.search(r'\{.*"highlights".*\}', generated_text, re.DOTALL)
    if not m:
        return -1.0
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return -1.0
    highlights = obj.get("highlights", [])
    if not isinstance(highlights, list) or len(highlights) == 0:
        return 0.0
    first = highlights[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return float(first["hook_score"])
        except (TypeError, ValueError):
            return -1.0
    return -1.0


def infer_one(model: Any, processor: Any, sample: dict, max_frames: int = MAX_FRAMES,
              base_dir: Optional[str] = None, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """1샘플 추론 -> {hook_score, raw, target}.

    target은 정답(학습 라벨)으로, 추론값과 즉석 비교용. 타임스탬프는 여기서
    다루지 않으며 호출부가 클립 윈도우로 재구성한다.
    """
    import torch

    inputs, parsed = build_inference_inputs(processor, sample, max_frames, base_dir)
    # 42일차: device_map 분산 모델 -> 입력은 GPU 진입점(cuda)에 둔다.
    #   accelerate 훅이 forward 중 CPU 오프로드 레이어로 자동 이동 처리.
    in_dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(in_dev) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "hook_score": parse_hook_score(text),
        "raw": text.strip(),
        "target": parsed["target"],
    }


def _load_first_sample(data_file: str) -> dict:
    """all.jsonl 첫 행을 dict로 로드(단독 실행 테스트용)."""
    with open(data_file, encoding="utf-8") as f:
        return json.loads(f.readline())


if __name__ == "__main__":
    import time

    DATA_FILE = "datasets/gemma_audio/all.jsonl"  # 프로젝트 루트 기준 상대경로
    sample = _load_first_sample(DATA_FILE)

    print("=== 모델 로드 중 (베이스 11GB + LoRA 어댑터) ===")
    t0 = time.time()
    model, processor = load_gemma()
    print(f"로드 완료 ({time.time() - t0:.1f}s)")

    print("=== 1샘플 추론 ===")
    t1 = time.time()
    result = infer_one(model, processor, sample)
    print(f"추론 완료 ({time.time() - t1:.1f}s)")

    print("-" * 40)
    print("hook_score :", result["hook_score"])
    print("raw 출력   :", result["raw"])
    print("정답 target:", result["target"])
    print("-" * 40)

    # 42일차: 관문 a - 최대 VRAM 사용량 실측(12GB 안에 들어가는지)
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"최대 VRAM 사용: {peak:.2f} GB / 12 GB")
    except Exception:
        pass