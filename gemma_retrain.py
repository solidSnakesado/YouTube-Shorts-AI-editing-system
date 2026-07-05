# 46일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_retrain.py
# [수정 1회·46일차] C(설정 비교): Qwen(붕괴없음)과 학습설정 일치. round9(JSON+graded)도
#   step300 붕괴 -> 라벨/형식 아닌 학습설정 차이 의심. Qwen train_qlora 대비 Gemma는
#   (a)MLP까지 LoRA (b)alpha/r=1 (c)rslora없음 (d)weight_decay0 (e)linear스케줄러 였음.
#   변경: finetune_mlp_modules False(attention만) · lora_alpha=rank*2 · use_rslora True
#   · weight_decay 0.01 · lr_scheduler cosine · rank기본8 · epochs기본3. Qwen과 전 변수
#   일치 -> 그래도 붕괴면 모델(E4B) 차이 확정. 변경 라인 아래 전달 메시지 참조.
#
# 45일차 수정(1회): --dropout 인자(기본 0). 이하 원본 이력:
# [수정 1회·45일차] --dropout 인자 추가(기본 0=기존동작 유지). 사유: B 검증 - round4가
#   step100서 건강한 분리(+0.44) 후 step200/300 붕괴 = 작은데이터 과적합 패턴. lora_dropout으로
#   과적합 억제해 초기 분리가 유지되는지 확인. 변경: build_model 시그니처+L(get_peft_model
#   lora_dropout)+인자+호출. 변경 라인 아래 전달 메시지 참조. ⚠️ dropout>0 시 Unsloth full
#   패치와 충돌 가능 - 학습 시작 직후 배너/trainable params 정상인지 확인할 것.
#
# 44일차 수정 | 이하 원본 이력:
# 수정3(안정화): L106~107 --save-limit 인자 · L158 save_total_limit 인자화(save_steps 작게 시 초반 best 보존).
# 수정 라인(본 파일 기준): L111 import(trl->transformers) · L140~161 config 블록
#   (SFTConfig->TrainingArguments · 주석추가 · warmup_steps · dataset_kwargs 삭제) · L163~170 trainer(SFTTrainer->Trainer)
# 사유: Unsloth가 SFTConfig에 push_to_hub_token 주입 -> TRL 1.7.0 거부 충돌 -> 순수 transformers로 회피
#
# 목적: baseline_r1 붕괴 -> round2 재학습. 핸드오프 4요건 중 본 스크립트는
#   (1) 체크포인트 재개(Drive) (2) eval 셋 연결(eval_loss) (3) A100 배치 스케일
#   (4) anti-collapse 설정 을 담는다. "붕괴 점수분리 모니터"는 학습중 생성
#   모드토글 리스크로 분리(다음 산출: gemma_collapse_check.py).
#
# anti-collapse 핵심:
#   - use_gradient_checkpointing="unsloth": Gemma4 E2B/E4B는 KV를 레이어간 공유
#     (num_kv_shared_layers). 일반 GC가 use_cache=False를 강제하면 KV공유 레이어가
#     로컬 재계산->logits 발산. Unsloth GC가 이 케이스를 처리(권장).
#   - label_smoothing OFF, target=데이터의 실제 hook_score(상대값), 응답-only 마스킹.
#   - lr/rank는 1차값 기본(인자로 조정) - 붕괴 재발 시 lr 하향/rank 상향 검토 knob.
#
# 의존(번들 추출본): gemma_collate.py, datasets/gemma_audio/{train,eval}.jsonl + frames/audio
# 실행(Colab A100, 마운트+추출 후):
#   python gemma_retrain.py
#   python gemma_retrain.py --rank 32 --lr 1e-4   # 붕괴 재발 시 knob 예시
# 끊김 시: 같은 명령 재실행 -> Drive 체크포인트 자동 감지하여 이어서 학습.
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# 44일차: 학습-추론 일치용 상수(gemma_inference와 동일: 불일치 시 결과 엉킴)
MAX_FRAMES = 8
MAX_SEQ_LENGTH = 3072
BASE_MODEL = "unsloth/gemma-4-E4B-it"          # bf16 원본(1차와 동일, bnb-4bit 아님)


def load_jsonl(path: str) -> list[dict]:
    """jsonl -> 행(dict) 리스트. 각 행은 {"messages":[user, assistant]} 구조."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """output_dir의 최신 checkpoint-* 경로(없으면 None). resume 자동감지용.

    1차는 체크포인트 0개라 끊기면 처음부터였음 -> 이 감지가 진짜 끊김 방어책.
    """
    if not os.path.isdir(output_dir):
        return None
    cks = [d for d in os.listdir(output_dir)
           if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
    if not cks:
        return None
    latest = max(cks, key=lambda d: int(d.split("-")[-1]))
    return os.path.join(output_dir, latest)


def build_model(rank: int, dropout: float = 0.0):
    """베이스(bf16) 로드 + LoRA 부착 -> (model, processor).

    vision 동결 + language LoRA만(1차와 동일, gemma_unsloth_check에서 검증된 인자).
    use_gradient_checkpointing="unsloth"로 KV공유 발산 회피. 부착 후 출력되는
    trainable params가 language-only 규모인지 확인할 것(audio까지 잡히면 VRAM 급증).
    45일차: dropout>0(B 검증)은 과적합 억제용. Unsloth 충돌 시 배너/params 이상으로 드러남.
    """
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,                             # 자동(bf16)
        load_in_4bit=False,                     # 1차와 동일 bf16 LoRA
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,           # vision/audio 동결
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,             # 46일차: Qwen과 동일 attention-only(MLP 끔)
        r=rank, lora_alpha=rank * 2,            # 46일차: Qwen과 동일 alpha/r=2
        lora_dropout=dropout,
        bias="none", random_state=3407,
        use_rslora=True,                        # 46일차: Qwen과 동일 rslora(학습 안정화)
        use_gradient_checkpointing="unsloth",   # KV공유 발산 방지(Unsloth 권장)
    )
    return model, processor


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma4 E4B round2 재학습(anti-collapse)")
    ap.add_argument("--train", default="datasets/gemma_audio/train.jsonl")
    ap.add_argument("--eval", default="datasets/gemma_audio/eval.jsonl")
    ap.add_argument("--output-dir",
                    default="/content/drive/MyDrive/gemma4_adapters/round2_ckpt",
                    help="Drive 경로 권장 - 런타임 죽어도 체크포인트 생존")
    ap.add_argument("--base-dir", default=None, help="미디어 상대경로 기준(보통 None=cwd)")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank(46일차: Qwen과 동일 8)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="LoRA dropout(B 검증: 과적합 억제 0.1~0.15, 0=기존동작)")
    ap.add_argument("--lr", type=float, default=2e-4, help="학습률(붕괴 시 1e-4 검토)")
    ap.add_argument("--batch", type=int, default=2, help="per-device 배치(1차=1, 보수적 2부터)")
    ap.add_argument("--accum", type=int, default=4, help="누적(batch2*accum4=유효8, 1차와 동일)")
    ap.add_argument("--epochs", type=float, default=3.0, help="에폭(46일차: Qwen과 동일 3)")
    ap.add_argument("--save-steps", type=int, default=100, help="체크포인트 주기(조기 검증 시 50)")
    ap.add_argument("--save-limit", type=int, default=3,
                    help="44일차: checkpoint 보존 개수(save_steps 작게 시 초반 best 보존 위해 늘림)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=256,
                    help="eval 부분집합(속도용, 0=전체 875). eval_loss는 보조지표")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    from gemma_collate import build_collate_fn

    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)
    if args.eval_samples > 0:
        eval_rows = eval_rows[: args.eval_samples]
    print(f"train {len(train_rows)} / eval {len(eval_rows)} 로드")

    # 44일차: 중첩 messages의 Arrow 스키마 추론 회피 -> 행을 JSON 문자열 단일 컬럼으로 보관
    train_ds = Dataset.from_dict(
        {"sample": [json.dumps(r, ensure_ascii=False) for r in train_rows]})
    eval_ds = Dataset.from_dict(
        {"sample": [json.dumps(r, ensure_ascii=False) for r in eval_rows]})

    model, processor = build_model(args.rank, args.dropout)

    # 44일차: 검증된 학습 collate 재사용. 래퍼가 JSON 문자열을 행 dict로 복원해 전달.
    base_collate = build_collate_fn(
        processor, max_frames=MAX_FRAMES, max_length=MAX_SEQ_LENGTH,
        base_dir=args.base_dir, truncate=False)

    def collate(batch: list[dict]) -> dict:
        return base_collate([json.loads(b["sample"]) for b in batch])

    ckpt = latest_checkpoint(args.output_dir)
    print(f"resume: {ckpt or '없음(처음부터)'}")

    # 44일차 수정: SFTConfig/SFTTrainer -> 순수 transformers. Unsloth 패치가 끼워넣는
    #   push_to_hub_token을 TRL 1.7.0 SFTConfig가 거부하는 충돌 회피. collator가 전처리를
    #   전담하므로 SFT 데이터셋 매직 불필요 -> Trainer로 충분(모델 GC 등 최적화는 유지).
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",             # 46일차: Qwen과 동일 cosine
        warmup_ratio=0.1,                       # 46일차: Qwen과 동일 warmup_ratio 0.1
        weight_decay=0.01,                      # 46일차: Qwen과 동일 L2 정규화(붕괴 방지 핵심)
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,       # 44일차: 인자화(초반 best 보존)
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        label_smoothing_factor=0.0,             # anti-collapse: OFF
        remove_unused_columns=False,            # 멀티모달 collator 필수
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        args=args_tr,
    )
    trainer.train(resume_from_checkpoint=ckpt)

    # 44일차: 체크포인트와 별개로 최종 어댑터 명시 저장(GGUF 변환 입력)
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"최종 어댑터 저장: {final_dir}")


if __name__ == "__main__":
    main()