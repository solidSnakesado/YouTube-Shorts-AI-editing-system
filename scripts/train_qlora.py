# 계층: 스크립트 (CLI 진입점)
# 역할: Unsloth QLoRA 파인튜닝 실행 - Qwen2.5-VL-7B 멀티모달 하이라이트 판별
# 의존: config.py, unsloth, transformers, trl, datasets, Pillow
# 22일차 신규: dataset.jsonl -> QLoRA 학습 -> LoRA 어댑터 저장
#   Gemma 4 E4B -> Qwen2.5-VL-7B 전환 (12GB VRAM 대응)
# 31일차: Phase 2 --max-seq-length CLI 추가 (10프레임 클립 대응), label_smoothing_factor + weight_decay 추가 (과적합 방지)
#
# 실행 방법: uv run python -m scripts.train_qlora [--adapter-type classifier|generator]
# 사전 설치: uv pip install unsloth trl datasets pillow

"""
QLoRA 파인튜닝 스크립트 - Qwen2.5-VL-7B 멀티모달 하이라이트 판별 학습

히트맵 기반 dataset.jsonl(21일차 수집)을 입력으로 받아
이미지 + 텍스트 -> "하이라이트"/"일반" 분류 태스트를 QLoRA로 학습
Unsloth FastVisionModel 사용 (12GB VRAM 에서 4bit QLoRA ~5GB)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings
from scripts.train_data_loader import load_dataset_jsonl, build_conversation_format

def run_training(
    dataset_path: Path, output_dir: Path, epochs: int = 3, batch_size: int = 1, grad_accum: int = 4,
    learning_rate: float = 2e-4, lora_r: int = 8, lora_alpha: int = 16, max_seq_length: int = 2048, 
    seed: int = 42, resume: bool = False,
) -> None:
    """QLoRA 파인튜닝 실행"""

    # 1) 데이터 로드
    raw_samples = load_dataset_jsonl(dataset_path)
    if len(raw_samples) < 10:
        raise ValueError(f"샘플 수 부족: {len(raw_samples)}개 (최소 10개 필요)")
    
    conversations = build_conversation_format(raw_samples)

    # 2) 학습/검증 분할 (80:20)
    import random
    random.seed(seed)
    random.shuffle(conversations)
    split_idx = max(1, int(len(conversations) * 0.8))
    train_data = conversations[:split_idx]
    eval_data = conversations[split_idx:]
    logger.info(f"분할: 학습 {len(train_data)}개, 검증 {len(eval_data)}개 (seed={seed} 셔플)")

    # 3) 모델 로드 (Unsloth 4bit)
    logger.info("Unsloth 모델 로드 시작 (4bit 양자화)...")

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=settings.LORA_BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=max_seq_length,
        use_gradient_checkpointing="unsloth"        # Unsloth 최적화 gradient checkpointing
    )

    # 4) LoRA 어댑터 적용 (VRAM 절약: r=8, 핵심 4개 모듈만)
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        lora_dropout=0.0,       # Unsloth 최적화: dropout=0 필수, 0 이외의 값이면 LoRA 레이어 최적화 패칭이 생략되어 tokenizer 초기화 실패
        use_rslora=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA 파라미터: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # 5) 데이터셋 구성 - pyarrow 우회, conversations 리스트 직접 사용
    # Dataset.from_dict/from_list는 content 길이 불일치로 pyarrow 에러 발생
    # skip_prepare_dataset=True 설정 시 SFTTrainer가 리스트를 직접 수용
    train_ds = train_data
    eval_ds = eval_data

    # 6) 학습 설정
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # label_smoothing_factor=0.05,      # 31일차: 과적합 방지 - 라벨 노이즈 내성 향상, 으로 추가 했지만 모델 붕괴 원인으로 비활성화 
        weight_decay=0.01,                  # 31일차: L2 정규화 - 파라미터 크기 제한
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        max_seq_length=max_seq_length,
        dataset_text_field="",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},      # pyarrow 스키마 변환 우회
        report_to="none",
    )

    # 7) 트레이너 구성 + 학습 실행
    FastVisionModel.for_training(model)                     # 학습 모드 활성화 (필수)

    # Qwen2.5-VL: from_pretrained 반환값은 processor 객체
    # SFTTrainer는 processing_class로 전달, UnslothVisionDataCollator는 processor 전달
    actual_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=actual_tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer)
    )

    logger.info(
        f"학습 시작 | 에폭: {epochs} | 배치: {batch_size} | "
        f"grad_accum: {grad_accum} | lr: {learning_rate}"
    )
    # resume=True 시 최신 체크포인트에서 이어서 학습
    resume_checkpoint = None
    if resume:
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
            if ckpts:
                resume_checkpoint = str(ckpts[-1])
                logger.info(f"체크포인트 재개: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 8) LoRA 어댑터 저장
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"LoRA 어댑터 저장 완료: {adapter_dir}")

    # 9) 학습 결과 요약
    log_history = trainer.state.log_history
    train_losses = [l["loss"] for l in log_history if "loss" in l]
    eval_losses = [l["eval_loss"] for l in log_history if "eval_loss" in l]

    summary = {
        "dataset_path": str(dataset_path),
        "total_samples": len(raw_samples),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "epochs": epochs,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "adapter_dir": str(adapter_dir),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"학습 완료 | train_loss: {summary['final_train_loss']:.4f} | "
        f"eval_loss: {summary['final_eval_loss']:.4f}"
    )

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA 파인튜닝 (Qwen2.5-VL-7B)")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help=f"dataset.jsonl 경로 (기본: {settings.FINETUNE_OUTPUT_DIR}/dataset.jsonl)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"출력 디렉토리 (기본: {settings.LORA_OUTPUT_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=3, help="에폭 수 (기본: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기 (기본: 1)")
    parser.add_argument("--grad-accum", type=int, default=4, help="그래디언트 누적 (기본: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="학습률 (기본: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (기본: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (기본: 16)")
    parser.add_argument("--adapter-type", type=str, default="classifier", choices=["classifier", "generator"], help="어댑터 타입 (기본: classifier)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="최대 시퀀스 길이 (기본: 2048, Phase 2 클립: 4096 권장)")
    parser.add_argument("--resume", action="store_true", help="최신 체크포인트에서 이어서 학습")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else (
        settings.finetune_output_path / "dataset.jsonl"
    )
    if args.output:
        output_dir = Path(args.output) 
    elif args.adapter_type == "generator":
        output_dir = Path(settings.LORA_OUTPUT_DIR).parent / "heatmap_generator"
    else:
        output_dir = Path(settings.LORA_OUTPUT_DIR)

    logger.info(f"데이터셋: {dataset_path}")
    logger.info(f"어댑터 타입: {args.adapter_type}")
    logger.info(f"출력 경로: {output_dir}")

    run_training(
        dataset_path=dataset_path,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()