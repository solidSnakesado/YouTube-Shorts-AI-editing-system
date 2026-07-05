# 45일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_audio_ablation.py
#
# [수정 1회] ablation 데이터 binclass -> relabel(JSON 지시문). 사유: base가 분류
#   지시문("'상'/'하'만")에선 입력무관 '상'만 출력(쏠림) -> 오디오 영향 측정 불가.
#   JSON 지시문에선 base가 다양 출력(6/21 확인) -> 오디오 유/무로 JSON 변화가 오염 없이 보임.
#   변경: --jsonl 기본 relabel, --max-new 64, 설명 갱신. 변경 라인 아래 전달 메시지 참조.
#
# 목적: (b) 입력 신호 검증 2단계. RMS 점검 결과 무음 거의 없음(0.4%)이나 pos/neg
#   에너지 분포 겹침(d=0.017) -> 에너지론 판별 불가. 단 RMS는 내용 무관(음색/패턴 못 봄).
#   이 스크립트는 "모델이 오디오 내용을 실제로 쓰는가"를 ablation으로 직접 확인.
#
#   방법: 동일 클립을 (A) 오디오 원본 / (B) 오디오 무음(zeros) 으로 각각 base 추론.
#     - 출력이 갈리면 -> 모델이 오디오를 봄(오디오는 쓰는데 분리 안 되는 다른 문제로).
#     - 출력이 같으면 -> 모델이 오디오 무시(시각만 판단). 동결 인코더 병목 증거.
#
#   왜 base(어댑터 없이): round2 checkpoint는 이미 붕괴(거의 전부 하)라 오디오 빼도
#     하만 나옴 -> 오디오 무시인지 붕괴 탓인지 구분 불가. base는 다양 출력(붕괴 아님)이라
#     오디오 유/무 영향이 편향 없이 보임.
#
# 재사용: load_images/load_audio/parse_sample/sample_frames(gemma_collate),
#   MAX_FRAMES(gemma_collapse_check). 생성 부분만 자체 구현(audio 인자 교체 위해).
#
# 의존(번들): gemma_collate.py + gemma_collapse_check.py.
# 실행(Colab A100, 학습 셀 정지 후):
#   python gemma_audio_ablation.py --n 12
#   python gemma_audio_ablation.py --n 12 --max-new 8
from __future__ import annotations

import argparse
import json
from typing import Optional

from gemma_collapse_check import MAX_FRAMES, BASE_MODEL
from gemma_collate import load_audio, load_images, parse_sample, sample_frames


def load_base(base_model: str):
    """베이스(bf16, 어댑터 없이) + 프로세서 로드. PeftModel 미부착."""
    import torch
    import transformers
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def infer_once(model, processor, parsed: dict, base_dir: Optional[str],
               mute: bool, max_new: int) -> str:
    """1회 추론. mute=True면 오디오를 같은 길이 무음(zeros)으로 대체."""
    import numpy as np
    import torch

    frames = sample_frames(parsed["frame_paths"], MAX_FRAMES)
    blocks: list[dict] = [{"type": "image"} for _ in frames]
    blocks.append({"type": "audio"})
    blocks.append({"type": "text", "text": parsed["instruction"]})
    text = processor.apply_chat_template(
        [{"role": "user", "content": blocks}], tokenize=False, add_generation_prompt=True)

    images = load_images(frames, base_dir)
    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)
    if mute:
        wav = np.zeros_like(wav)                    # 길이 동일 무음(인코더엔 들어가되 신호 0)

    inputs = processor(text=[text], images=[images], audio=[wav],
                       return_tensors="pt", padding=True,
                       truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    plen = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def select(jsonl: str, n: int):
    """pos n / neg n 선택(metadata.label 기준)."""
    pos: list[dict] = []
    neg: list[dict] = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            is_pos = row.get("metadata", {}).get("label") != "negative"
            if is_pos and len(pos) < n:
                pos.append(row)
            elif (not is_pos) and len(neg) < n:
                neg.append(row)
            if len(pos) >= n and len(neg) >= n:
                break
    return pos, neg


def run(model, processor, rows, base_dir, max_new, tag):
    """각 행을 오디오 유/무로 추론 -> 변경 여부 집계."""
    changed = 0
    for i, row in enumerate(rows):
        parsed = parse_sample(row)
        on = infer_once(model, processor, parsed, base_dir, False, max_new)
        off = infer_once(model, processor, parsed, base_dir, True, max_new)
        diff = on != off
        changed += int(diff)
        mark = "변함" if diff else "동일"
        print(f"  {tag}[{i}] {mark}: 오디오있음={on!r} | 무음={off!r}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="오디오 ablation(base + JSON 지시문, relabel)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--n", type=int, default=12, help="pos/neg 각 표본 수")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--max-new", type=int, default=64, help="생성 토큰 수(JSON 출력이라 충분히)")
    args = ap.parse_args()

    pos_rows, neg_rows = select(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)}")
    print(f"=== base 로드: {args.base_model} (어댑터 없이) ===")
    model, processor = load_base(args.base_model)

    print("=== pos: 오디오 유/무 비교 ===")
    pc = run(model, processor, pos_rows, args.base_dir, args.max_new, "pos")
    print("=== neg: 오디오 유/무 비교 ===")
    nc = run(model, processor, neg_rows, args.base_dir, args.max_new, "neg")

    total = len(pos_rows) + len(neg_rows)
    chg = pc + nc
    print("-" * 56)
    print(f"오디오 유/무로 출력 변한 비율: {chg}/{total} ({chg / total:.1%}) "
          f"(pos {pc}/{len(pos_rows)}, neg {nc}/{len(neg_rows)})")
    print("-" * 56)
    if total == 0:
        print("해석: 표본 없음")
    elif chg / total < 0.2:
        print("해석: 대부분 동일 -> 모델이 오디오를 사실상 무시(시각만 판단). "
              "동결 오디오 인코더가 병목 증거. 다음: 인코더 일부 학습 검토.")
    elif chg / total > 0.6:
        print("해석: 다수 변함 -> 모델이 오디오 내용을 봄. 오디오는 쓰는데 분리 안 되는 "
              "다른 문제(인코더 표현력/라벨 품질)로. ablation으론 오디오 무시 아님.")
    else:
        print("해석: 부분적 변함 -> 오디오 일부 반영. 위 사례별로 직접 판단.")


if __name__ == "__main__":
    main()