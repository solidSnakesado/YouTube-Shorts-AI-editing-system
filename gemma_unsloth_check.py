# 44일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_unsloth_check.py
#
# 목적: 재학습을 unsloth로 갈지 unsloth-free(transformers+peft+TRL)로 갈지 첫 판단.
#   43일차 핸드오프 경고 - Colab 기본 torch 2.11+cu128 / transformers 5.12 가
#   unsloth 허용 범위(torch<2.11)를 초과 -> 설치/임포트/로드가 깨질 수 있음.
#   (1)import (2)베이스 로드 (3)LoRA 부착 까지 실제로 돌려보고 결정한다.
#
# 선행(셀에서 먼저 실행, 런타임 끊기면 매번):
#   !pip install -q --no-deps unsloth unsloth_zoo bitsandbytes trl peft accelerate
#   !pip install -q "torchao>=0.16" soundfile librosa
#   (torchao가 런타임 재시작 요구하면 1회 재시작 후 이 스크립트 실행)
# 실행: python gemma_unsloth_check.py             # bf16 (1차 학습과 동일 경로)
#       python gemma_unsloth_check.py --load-4bit  # 4bit 빠른 스모크
#
# 판정: 3단계 모두 OK -> unsloth 사용 가능(재학습 셀 진행)
#       import/load 실패 -> unsloth-free 폴백 권장(gemma_native_check 경로가 이미 검증됨)
# 주의: 추론/학습 스텝 없음. 로드까지만(버전 호환 증명). bf16 모델 다운로드(~16GB) 시간 별도.
from __future__ import annotations

import argparse
import importlib
import platform


def _v(mod_name: str) -> str:
    """44일차: 모듈 버전 문자열 안전 조회(없으면 미설치/오류 표기)."""
    try:
        m = importlib.import_module(mod_name)
        return getattr(m, "__version__", "??")
    except Exception as e:                          # noqa: BLE001 - 미설치도 진단 정보
        return f"미설치/오류({type(e).__name__})"


def print_env() -> None:
    """44일차: torch/transformers/unsloth 등 버전 + GPU 출력 - 충돌 진단 근거."""
    print("=== 환경 버전 ===")
    print(f"  python       : {platform.python_version()}")
    for name in ("torch", "transformers", "unsloth", "unsloth_zoo",
                 "peft", "trl", "bitsandbytes", "timm", "torchao"):
        print(f"  {name:13s}: {_v(name)}")
    try:
        import torch
        print(f"  CUDA 사용가능 : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    except Exception as e:                          # noqa: BLE001
        print(f"  torch CUDA 조회 실패: {type(e).__name__}: {e}")


def step_import() -> bool:
    """44일차 [1/3]: unsloth 임포트. 버전 어서션이 여기서 터지면 즉시 폴백 신호."""
    print("\n=== [1/3] unsloth import ===")
    try:
        from unsloth import FastModel
        print(f"  OK: from unsloth import {FastModel.__name__}")
        return True
    except Exception as e:                          # noqa: BLE001
        print(f"  실패: {type(e).__name__}: {e}")
        return False


def step_load(model_name: str, load_4bit: bool):
    """44일차 [2/3]: 베이스 로드. E계열 timm 오탐(mobilenetv5) 시 use_exact 폴백.

    멀티모달이라 from_pretrained 는 (model, processor) 반환.
    bf16(load_in_4bit=False)이 1차 학습과 동일 경로 - VRAM 15.4GB 기록과 일치.
    """
    print(f"\n=== [2/3] 베이스 로드 ({'4bit' if load_4bit else 'bf16'}) ===")
    from unsloth import FastModel
    kw = dict(model_name=model_name, max_seq_length=3072,
              load_in_4bit=load_4bit, full_finetuning=False)
    if not load_4bit:
        kw["dtype"] = None                          # 자동(bf16)
    try:
        model, processor = FastModel.from_pretrained(**kw)
        print(f"  OK: {model_name} 로드 (processor={type(processor).__name__})")
        return model, processor
    except Exception as e:                          # noqa: BLE001 - E계열 오탐은 use_exact 재시도
        print(f"  1차 시도 실패: {type(e).__name__}: {e}")
        print("  -> use_exact_model_name=True 로 재시도")
        try:
            model, processor = FastModel.from_pretrained(
                use_exact_model_name=True, **kw)
            print(f"  OK(재시도): {model_name} 로드")
            return model, processor
        except Exception as e2:                     # noqa: BLE001
            print(f"  재시도도 실패: {type(e2).__name__}: {e2}")
            return None, None


def step_peft(model) -> bool:
    """44일차 [3/3]: LoRA 부착(학습 셋업 경로). vision 동결 + language LoRA(r16).

    학습 스텝은 없음 - get_peft_model 이 현재 버전에서 도는지까지만 검증.
    동결/타겟 세부는 재학습 셀에서 확정(여기선 1차와 동일 r16 기준).
    """
    print("\n=== [3/3] LoRA 부착(get_peft_model) ===")
    from unsloth import FastModel
    try:
        FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16, lora_alpha=16, lora_dropout=0,
            bias="none", random_state=3407,
        )
        print("  OK: LoRA 어댑터 부착(r16, vision 동결)")
        return True
    except Exception as e:                          # noqa: BLE001
        print(f"  실패: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="unsloth 설치/로드 검증(재학습 프레임워크 결정)")
    ap.add_argument("--model", default="unsloth/gemma-4-E4B-it",
                    help="베이스 모델(gemma_config.GEMMA_BASE_MODEL과 동일)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4bit 빠른 스모크(기본=bf16, 1차 학습과 동일 경로)")
    args = ap.parse_args()

    print_env()

    if not step_import():
        print("\n판정: unsloth import 실패 -> unsloth-free 폴백 권장")
        print("  (gemma_native_check.py 의 transformers+peft 경로가 이미 검증됨)")
        return

    model, _proc = step_load(args.model, args.load_4bit)
    if model is None:
        print("\n판정: 베이스 로드 실패 -> unsloth-free 폴백 권장")
        return

    ok_peft = step_peft(model)
    print("\n" + "=" * 56)
    if ok_peft:
        print("판정: 3단계 모두 OK -> unsloth 사용 가능(재학습 셀 진행)")
    else:
        print("판정: 로드 OK / LoRA 부착 실패 -> 로그 확인 후 unsloth-free 검토")


if __name__ == "__main__":
    main()