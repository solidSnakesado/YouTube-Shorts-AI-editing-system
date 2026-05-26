# 계층: 비즈니스 로직 계층 (Service)
# 역할: LoRA 모델 로드/언로드/추론/프레임 변환 공통 유틸
# 분리 출처: vlm_client.py (25일차 - 300줄 한계 대응)
# 의존: unsloth, transformers, PIL, torch (지연 import)

"""LoRA 공통 유틸 - 모델 생명주기 + 추론 헬퍼"""

import gc
from loguru import logger

# --------------------------------------------------------------
# 모델 로드 / 언로드
# --------------------------------------------------------------

def load_lora_model(adapter_path: str):
    """
    LoRA 어댑터 로드 (Unsloth FastVisionModel, 4bit).

    Returns:
        (model, tokenizer, processor) 튜플
    """

    from unsloth import FastVisionModel
    from transformers import AutoProcessor

    logger.info(f"LoRA 모델 로드: {adapter_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=True,
        max_seq_length=4096,
    )
    FastVisionModel.for_inference(model)
    processor = AutoProcessor.from_pretrained(adapter_path)
    return model, tokenizer, processor

def unload_lora_model(model, tokenizer, processor) -> None:
    """LoRA 모델 VRAM 해제 (del + gc + cuda empty_cache)"""

    import torch

    del model, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("LoRA 모델 VRAM 해제 완료")

# --------------------------------------------------------------
# 프레임 변환
# --------------------------------------------------------------

def frames_to_pil(frames: list[dict], max_count: int = 3) -> list:
    """
    base64 인코딩 프레임 리스트 -> PIL Image 리스트 변환

    Args:
        frames: [{"base64": str, ...}, ...]
        max_count: 변환할 최대 프레임 수 (VRAM 절약)

    Returns:
        PIL.Image 리스트
    """

    import base64
    import io
    from PIL import Image

    result = []
    for f in frames[:max_count]:
        try:
            data = base64.b64decode(f["base64"])
            result.append(Image.open(io.BytesIO(data)))
        except Exception as e:
            logger.warning(f"프레임 변환 실패 (건너뜀): {e}")
    return result

# --------------------------------------------------------------
# 추론
# --------------------------------------------------------------

def lora_generate(model, tokenizer, processor, messages: list[dict], max_tokens: int = 1024, temp: float = 0.3) -> str:
    """
    LoRA 모델로 텍스트 생성.
    입력 토큰을 제거하고 생성된 텍스트만 반환

    Args:
        messages: [{"role": "user", "content": [...]}]
        max_tokens: 최대 생성 토큰 수
        temp: 샘플링 온도 (0이면 greedy)

    Returns:
        생성된 텍스트 문자열
    """

    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp, do_sample=(temp > 0))

    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)