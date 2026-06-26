# 43일차 신규: Gemma 어댑터 네이티브(transformers) 추론 검증 — 붕괴(어댑터) vs GGUF변환 판별
# 배치 경로: 저장소 루트 ~/project/yt_shorts_ai/gemma_native_check.py (Colab A100 실행, 신규)
#
# 배경: GGUF(llama.cpp) 추론에서 pos(gt 0.9551)·neg(gt [])이 동일 상수 0.8229 -> 붕괴.
#       단 GGUF 변환 자체가 원인일 가능성을 최종 배제하려면 학습 환경에서 직접 확인해야 한다.
#       학습 때 쓴 transformers 스택(gemma_inference.load_gemma/infer_one)으로 동일 어댑터를
#       동일 pos/neg 샘플에 greedy 8프레임 추론한다(GGUF와 같은 조건: do_sample=False, 8프레임).
#   - 네이티브가 pos>neg 로 구별(또는 neg=[]) -> GGUF 변환이 원인 -> 변환(머지) 수정, 재학습 불필요
#   - 네이티브도 동일 상수 -> 어댑터 붕괴(H1) 확정 -> 재학습(anti-collapse 적용)
#
# ⚠️ 로컬 12GB 불가(transformers 4bit가 21.97GB). A100 필수. 재학습 세션 첫 셀로 실행 권장.
# 의존: gemma_inference.py, gemma_mtmd_probe.py, gemma_collate.py (모두 저장소 루트)
# 사용(Colab): python gemma_native_check.py   (또는 노트북에서 main() 호출)
from __future__ import annotations

import argparse

from gemma_collate import parse_sample
from gemma_mtmd_probe import find_row
from gemma_inference import load_gemma, infer_one

DATA = "datasets/gemma_audio/all.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Gemma 어댑터 네이티브 추론 검증(pos vs neg)")
    ap.add_argument("--jsonl", default=DATA)
    ap.add_argument("--base-dir", default=None, help="미디어 경로 접두(상대경로 보정)")
    args = ap.parse_args()

    # 43일차: GGUF 스윕과 동일한 방식으로 pos/neg 샘플 1개씩 선택(정답 라벨 기준)
    _lp, pos_row, _mp = find_row(args.jsonl, None, "pos")
    _ln, neg_row, _mn = find_row(args.jsonl, None, "neg")
    pos_t = parse_sample(pos_row)["target"]
    neg_t = parse_sample(neg_row)["target"]

    print("=== 어댑터 로드(베이스 + LoRA, A100) ===")
    model, processor = load_gemma()

    print("=== pos 추론(8프레임 greedy) ===")
    pr = infer_one(model, processor, pos_row, base_dir=args.base_dir)
    print("=== neg 추론(8프레임 greedy) ===")
    nr = infer_one(model, processor, neg_row, base_dir=args.base_dir)

    ps, ns = pr["hook_score"], nr["hook_score"]
    print("-" * 56)
    print(f"pos  정답={pos_t}")
    print(f"     추론 hook_score={ps}  raw={pr['raw']}")
    print(f"neg  정답={neg_t}")
    print(f"     추론 hook_score={ns}  raw={nr['raw']}")
    print("-" * 56)
    print("(참고: GGUF 결과는 pos=neg=0.8229 상수였음)")

    if ps is None or ns is None:
        print("판정: 추론 파싱 실패(-1.0/None) 포함 -> 위 raw 출력으로 직접 판단")
        return
    print(f"구별(pos-neg) = {ps - ns:+.4f}")
    if ns == 0.0 or (ps - ns) > 0.10:
        print("판정: 네이티브가 구별함 -> GGUF 변환이 원인 -> 변환(머지 등) 수정 (재학습 불필요)")
    elif abs(ps - ns) < 0.05:
        print("판정: 네이티브도 거의 상수 -> 어댑터 붕괴(H1) 확정 -> 재학습(anti-collapse)")
    else:
        print("판정: 애매(부분 구별) -> 위 raw 출력으로 직접 판단")


if __name__ == "__main__":
    main()