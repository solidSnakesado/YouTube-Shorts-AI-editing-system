# 44일차 수정 | 배치: ~/project/yt_shorts_ai/gemma_collapse_check.py
# 수정3(가-숫자형식): L32~60 parse_hook_score 순수 숫자 우선+JSON 폴백(출력형식 재설계 대응).
# 수정 라인(본 파일 기준): L109~128 select_samples(label/임계 판정 - 재라벨 neg 점수>0 대응) ·
#   L160·L167 추론 헤더 문구 · L150 --pos-threshold · L187~201 판정(방향정확+분리도 게이트)
# 사유: neg 재라벨로 neg도 점수>0 -> gt==0=neg 판정 깨짐. label 우선/임계로 교체 + 회귀 판정 적응.
#
# 목적: 재학습 중 Drive에 쌓이는 checkpoint를 검사해 붕괴 여부를 fail-fast 판정.
#   pos/neg 각 N개를 추론 -> 점수 분포 + 분리도(gap) 출력 -> 계속/중단 결정.
#   loss는 붕괴를 못 잡음(1차 0.04인데 붕괴) -> 점수 분포가 1차 지표.
#
# 로딩: 학습(unsloth)과 별도 프로세스 -> unsloth 불필요. transformers
#   AutoModelForImageTextToText(bf16) + peft(checkpoint 어댑터). GPU 여유(학습33+검사16<80).
#   전처리는 gemma_collate 재사용(학습-추론 일치). gemma_inference 미의존(번들 외).
#
# 의존(번들): gemma_collate.py 만. (본 파일 gemma_collapse_check.py 만 추가 업로드)
# 실행(학습과 다른 셀, checkpoint 저장 완료 후):
#   python gemma_collapse_check.py --checkpoint /content/drive/MyDrive/gemma4_adapters/round2_ckpt/checkpoint-100
#   python gemma_collapse_check.py --checkpoint <경로> --n 8   # 표본 늘리기
from __future__ import annotations

import argparse
import json
import re
from typing import Optional

from gemma_collate import load_audio, load_images, parse_sample, sample_frames

BASE_MODEL = "unsloth/gemma-4-E4B-it"   # 44일차: bf16 원본 강제(config의 bnb-4bit면 bf16 로드 AssertionError 회피)
MAX_FRAMES = 8                          # 학습 max_frames=8과 동일(불일치 시 결과 엉킴)
MAX_NEW = 64


def parse_hook_score(text: str) -> float:
    """생성/타겟 텍스트 -> hook_score. 44일차(가): 순수 숫자 우선.
    "0.73"->0.73, 구버전 JSON({highlights:[{hook_score:x}]})도 폴백 호환, 실패->-1.0."""
    s = text.strip()
    # (가) 순수 숫자 형식 우선: 맨 앞 0~1 실수 추출(모델이 뒤에 잡소리 붙여도 첫 숫자)
    m = re.match(r'\s*(0?\.\d+|[01](?:\.\d+)?)', s)
    if m and '{' not in s[:m.start() + 1]:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            pass
    # 구버전 JSON 폴백
    mj = re.search(r'\{.*"highlights".*\}', text, re.DOTALL)
    if not mj:
        return -1.0
    try:
        obj = json.loads(mj.group(0))
    except json.JSONDecodeError:
        return -1.0
    hl = obj.get("highlights", [])
    if not isinstance(hl, list) or len(hl) == 0:
        return 0.0
    first = hl[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return float(first["hook_score"])
        except (TypeError, ValueError):
            return -1.0
    return -1.0


def load_checkpoint(adapter_dir: str, base_model: str):
    """베이스(bf16) + checkpoint LoRA(peft) 로드. unsloth 미사용(별도 프로세스)."""
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor

    cls = (getattr(transformers, "AutoModelForImageTextToText", None)
           or getattr(transformers, "AutoModelForMultimodalLM"))
    model = cls.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    try:
        processor = AutoProcessor.from_pretrained(adapter_dir, padding_side="left")
    except Exception:                               # checkpoint에 프로세서 없으면 베이스에서
        processor = AutoProcessor.from_pretrained(base_model, padding_side="left")
    return model, processor


def infer(model, processor, row: dict, base_dir: Optional[str]) -> tuple[float, str]:
    """1샘플 추론 -> (pred_hook, raw). 학습 collate와 동일 프레임 구성 + 생성 프롬프트."""
    import torch

    parsed = parse_sample(row)
    frames = sample_frames(parsed["frame_paths"], MAX_FRAMES)
    blocks: list[dict] = [{"type": "image"} for _ in frames]
    blocks.append({"type": "audio"})
    blocks.append({"type": "text", "text": parsed["instruction"]})
    text = processor.apply_chat_template(
        [{"role": "user", "content": blocks}], tokenize=False, add_generation_prompt=True)

    images = load_images(frames, base_dir)
    wav, _sr = load_audio(parsed["audio_path"], base_dir=base_dir)
    # 44일차: Gemma4 오디오 처리 - truncation=False/max_length 명시(아니면 max_length 없음 에러)
    inputs = processor(text=[text], images=[images], audio=[wav],
                       return_tensors="pt", padding=True,
                       truncation=False, max_length=480000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    plen = inputs["input_ids"].shape[1]
    gen = processor.tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    return parse_hook_score(gen), gen.strip()


def select_samples(jsonl: str, n: int, thr: float):
    """jsonl에서 pos n개 / neg n개 선택. 재라벨 후 neg도 점수>0이라 정답 기준을
    metadata.label 우선(없으면 hook_score 임계 thr: pos=label 외 or gt>=thr)으로 판정."""
    pos: list[dict] = []
    neg: list[dict] = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("metadata", {}).get("label")
            try:
                gt = parse_hook_score(parse_sample(row)["target"])
            except Exception:                       # noqa: BLE001 - 형식 이상 행은 건너뜀
                continue
            is_pos = (label != "negative") if label else (gt >= thr)
            if is_pos and len(pos) < n:
                pos.append(row)
            elif (not is_pos) and len(neg) < n:
                neg.append(row)
            if len(pos) >= n and len(neg) >= n:
                break
    return pos, neg


def _stats(scores: list[float]):
    """파싱 성공 점수만으로 (n, mean, min, max). 없으면 (0, None, None, None)."""
    ok = [s for s in scores if s >= 0]
    if not ok:
        return 0, None, None, None
    return len(ok), sum(ok) / len(ok), min(ok), max(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint 붕괴 검사(pos/neg 점수분리)")
    ap.add_argument("--checkpoint", required=True, help="검사할 checkpoint-* 경로")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/eval.jsonl")
    ap.add_argument("--n", type=int, default=5, help="pos/neg 각 표본 수")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-dir", default=None, help="미디어 상대경로 기준(보통 None=cwd)")
    ap.add_argument("--pos-threshold", type=float, default=0.5,
                    help="label 없을 때 pos/neg 판정 임계 + 방향정확 기준(재라벨: neg<0.5, pos>=0.8)")
    args = ap.parse_args()

    pos_rows, neg_rows = select_samples(args.jsonl, args.n, args.pos_threshold)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)} (요청 {args.n})")

    print(f"=== 로드: {args.checkpoint} (베이스 {args.base_model}, bf16) ===")
    model, processor = load_checkpoint(args.checkpoint, args.base_model)

    print("=== pos 추론(정답 hook 높음 ~0.8-1.0) ===")
    pos_pred: list[float] = []
    for i, row in enumerate(pos_rows):
        p, raw = infer(model, processor, row, args.base_dir)
        pos_pred.append(p)
        print(f"  pos[{i}] pred={p:+.4f}  raw={raw}")

    print("=== neg 추론(정답 hook 낮음 ~0.0-0.5) ===")
    neg_pred: list[float] = []
    for i, row in enumerate(neg_rows):
        p, raw = infer(model, processor, row, args.base_dir)
        neg_pred.append(p)
        print(f"  neg[{i}] pred={p:+.4f}  raw={raw}")

    pn, pmean, pmin, pmax = _stats(pos_pred)
    nn, nmean, nmin, nmax = _stats(neg_pred)
    print("-" * 56)
    print(f"pos: n={pn} mean={pmean} range=[{pmin}, {pmax}]")
    print(f"neg: n={nn} mean={nmean} range=[{nmin}, {nmax}]")

    if pmean is None or nmean is None:
        print("판정: 파싱 성공 표본 부족 -> 위 raw로 직접 판단")
        return

    all_ok = [s for s in pos_pred + neg_pred if s >= 0]
    total_spread = max(all_ok) - min(all_ok)
    sep = pmean - nmean
    thr = args.pos_threshold
    pos_hi = sum(1 for s in pos_pred if s >= thr)
    neg_lo = sum(1 for s in neg_pred if 0 <= s < thr)
    print(f"전체 분산={total_spread:.4f}  분리도(pos-neg)={sep:+.4f}  "
          f"방향정확(pos≥{thr}:{pos_hi}/{pn}, neg<{thr}:{neg_lo}/{nn})")
    print("-" * 56)

    if total_spread < 0.05:
        print(f"판정: 붕괴(입력무관 상수) — 전체가 {min(all_ok):.4f}~{max(all_ok):.4f}에 뭉침. 중단+knob 조정")
    elif sep >= 0.2 and pos_hi >= pn * 0.6 and neg_lo >= nn * 0.6:
        print(f"판정: 건강한 분리 — pos 높고 neg 낮음(gap {sep:+.4f}). 학습 계속 진행 가능")
    elif sep < 0.05:
        print("판정: 붕괴 의심 — pos/neg 구별 거의 없음. 중단+knob(rank↑/lr↓) 검토")
    else:
        print(f"판정: 애매(부분 분리, gap {sep:+.4f}) — 위 표/분포로 직접 판단")


if __name__ == "__main__":
    main()