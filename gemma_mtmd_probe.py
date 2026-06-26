# 43일차 신규: Gemma GGUF(llama-mtmd-cli) 8프레임 추론 명령 빌더 (과제 #1 검증 도구)
# 배치 경로: 저장소 루트 ~/project/yt_shorts_ai/gemma_mtmd_probe.py (독립 도구, 신규)
#
# 목적: all.jsonl 한 샘플에서 학습과 동일하게 N프레임을 균등 샘플링하고,
#       콤마구분 단일 --image + --audio 가 포함된 llama-mtmd-cli 명령을 그대로 출력한다.
#       42일차 1프레임 테스트가 부정 샘플(neg_240)을 hook_score 1.0 으로 오판 ->
#       학습과 같은 8프레임으로 재검증해 입력 방식을 확정하기 위함.
#       43일차 수정: 반복 --image 는 deprecated(마지막 값만 적용) -> 콤마구분 1개로 교체.
#
# 핵심: 프레임 균등 샘플링 + content-block 파싱을 학습 collate(gemma_collate)에서
#       그대로 재사용 -> 학습-추론 입력 일치 보장.
#       프롬프트도 기본값으로 샘플의 학습 지시문(instruction)을 사용한다(임의 문구 X).
#
# 사용:
#   python3 gemma_mtmd_probe.py --id neg_240 --n-frames 8      # 경로 부분문자열로 선택
#   python3 gemma_mtmd_probe.py --target pos --n-frames 8      # 정답 라벨로 긍정 샘플 선택
#   python3 gemma_mtmd_probe.py --target neg --base-dir datasets/gemma_audio
#   43일차 추가: 출력 명령에 --temp 0(greedy) 고정, --target(pos/neg) 정답 기준 선택
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from gemma_collate import parse_sample, sample_frames  # 43일차: 학습 전처리 재사용

# 42일차 핸드오프에서 확정된 GGUF 자산 경로 (models/gguf/gemma4/)
BIN = "./llama.cpp/build/bin/llama-mtmd-cli"
MODEL = "models/gguf/gemma4/gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ = "models/gguf/gemma4/mmproj-gemma-4-E4B-it-Q8_0.gguf"
LORA = "models/gguf/gemma4/baseline_r1_lora.gguf"
LORA_SCALE = "2.0"   # 42일차: 1.0 무반응 / 2.0 정상 / 5.0 과포화
N_PREDICT = 128


def target_is_pos(target_str: str):
    """43일차: 정답 JSON 문자열이 긍정(하이라이트 있음)인지 판정. 불명이면 None.

    {"highlights": []}                       -> False (부정)
    {"highlights": [{"hook_score": ...}]}    -> True  (긍정)
    """
    try:
        obj = json.loads(target_str)
    except json.JSONDecodeError:
        return None
    hl = obj.get("highlights", [])
    return isinstance(hl, list) and len(hl) > 0


def find_row(jsonl_path: str, sample_id=None, target_kind=None):
    """frame/audio 경로 부분문자열(sample_id) 또는 정답 라벨(target_kind: pos/neg)로 행 선택.

    둘 다 주면 둘 다 만족하는 행, 둘 다 없으면 첫 행을 사용한다.
    반환: (행번호(1-base), 행 dict, 매칭된 모든 행번호 리스트).
    """
    matches: list[int] = []
    first = None
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if sample_id and sample_id not in line:
                continue
            row = json.loads(line)
            try:
                parsed = parse_sample(row)
            except ValueError:
                continue
            if sample_id:
                paths = parsed["frame_paths"] + [parsed["audio_path"]]
                if not any(sample_id in p for p in paths):
                    continue
            if target_kind:                       # 43일차: 정답 라벨 필터
                is_pos = target_is_pos(parsed["target"])
                if is_pos is None or (target_kind == "pos") != is_pos:
                    continue
            matches.append(lineno)
            if first is None:
                first = (lineno, row)
    if first is None:
        sel = sample_id or (f"target={target_kind}" if target_kind else "(첫 행)")
        raise SystemExit(f"'{sel}' 조건의 샘플을 {jsonl_path} 에서 찾지 못함")
    return first[0], first[1], matches


def build_command(frames: list[str], audio: str, prompt: str,
                  base_dir: str | None) -> str:
    """샘플된 프레임/오디오/지시문으로 실행 가능한 llama-mtmd-cli 명령 생성."""
    def media(p: str) -> str:
        return str(Path(base_dir) / p) if base_dir else p

    lines = [
        BIN,
        f"  -m {MODEL}",
        f"  --mmproj {MMPROJ}",
        f"  --lora-scaled {LORA}:{LORA_SCALE}",
        "  -ngl 99 --jinja --temp 0",   # 43일차: greedy 고정(재현성) - 샘플링 노이즈 제거
    ]
    # 43일차: 반복 --image 는 deprecated(마지막 값만 적용) -> 콤마구분 단일 인자로 N프레임 전달
    imgs = ",".join(media(fr) for fr in frames)
    lines.append(f"  --image {shlex.quote(imgs)}")
    lines.append(f"  --audio {shlex.quote(media(audio))}")
    lines.append(f"  -p {shlex.quote(prompt)}")
    lines.append(f"  -n {N_PREDICT}")
    return " \\\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma GGUF N프레임 추론 명령 빌더")
    ap.add_argument("--id", default=None, help="frame/audio 경로 부분문자열로 샘플 선택")
    ap.add_argument("--target", choices=["pos", "neg"], default=None,
                    help="정답 라벨로 선택 (pos=하이라이트 있음 / neg=빈 리스트)")
    ap.add_argument("--n-frames", type=int, default=8, help="균등 샘플 프레임 수 (학습=8)")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/all.jsonl",
                    help="데이터셋 경로 (기본 datasets/gemma_audio/all.jsonl)")
    ap.add_argument("--base-dir", default=None, help="미디어 경로 접두(상대경로 보정)")
    ap.add_argument("--prompt", default=None,
                    help="프롬프트 강제 지정 (미지정 시 샘플의 학습 지시문 사용)")
    args = ap.parse_args()

    lineno, row, matches = find_row(args.jsonl, args.id, args.target)
    parsed = parse_sample(row)
    frames = sample_frames(parsed["frame_paths"], args.n_frames)
    prompt = args.prompt if args.prompt else parsed["instruction"]

    sel = args.id or (f"target={args.target}" if args.target else "(첫 행)")
    head = matches[:5]
    print(f"[행 {lineno}] {sel} 매칭 (전체 {len(matches)}개: {head}"
          f"{' ...' if len(matches) > 5 else ''})")
    if args.id and len(matches) > 1:
        print(f"  ⚠️ 매칭 {len(matches)}개 - 첫 행만 사용. --id 를 더 구체화 권장")
    print(f"  전체 프레임 {len(parsed['frame_paths'])}개 -> {len(frames)}개 균등 샘플")
    print(f"  오디오: {parsed['audio_path']}")
    print(f"  프롬프트: {'(학습 지시문)' if not args.prompt else '(강제)'} {prompt[:60]}...")
    print(f"  정답(target): {parsed['target']}")
    print("-" * 60)
    print(build_command(frames, parsed["audio_path"], prompt, args.base_dir))
    print("-" * 60)
    # 43일차: 기대 출력은 휴리스틱(id 접두) 대신 실제 정답 라벨로 판정
    if target_is_pos(parsed["target"]):
        print(f"기대 출력: {parsed['target']}  <- 긍정 샘플(hook_score 있어야 정답)")
    else:
        print('기대 출력: {"highlights": []}  <- 부정 샘플 정답(빈 리스트)')


if __name__ == "__main__":
    main()