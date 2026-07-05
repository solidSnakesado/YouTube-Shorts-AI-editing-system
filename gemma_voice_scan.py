# 45일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_voice_scan.py
#
# 목적: A단계 1차. transcript 분리도 측정 전, Gemma 데이터에 사람 음성이 있는 클립이
#   얼마나 되는지 Whisper로 스캔. 무음 게임플레이(효과음/BGM만)는 transcript가 빈칸이라
#   측정 불가 -> 음성 클립 비율을 먼저 확인해야 측정 진행 여부 결정 가능.
#
#   판별:
#     - 음성 클립 충분(pos/neg 각 ~30%+) -> 그 부분집합으로 transcript 분리도 측정 진행.
#     - 거의 없음(순수 무음 게임플레이) -> transcript 구조적 불가. "무음은 59% 본질적 한계"
#       확정. 라벨 아닌 타깃(무음 게임플레이)의 한계 결론.
#
# 방식: faster-whisper로 각 클립 오디오 전사 -> 텍스트 길이/세그먼트로 음성 유무 판정.
#   무음/효과음은 빈 텍스트 또는 매우 짧은 노이즈 -> MIN_CHARS 기준으로 음성 클립 카운트.
#   기존 dataset_transcriber 패턴(faster-whisper, VRAM 관리) 참고하되 독립 실행 가능하게 작성.
#
# 실행(로컬 WSL, RTX 5070 Ti로 충분 - A100 불필요):
#   python gemma_voice_scan.py --n 80
#   python gemma_voice_scan.py --n 80 --model large-v3-turbo   # 정확도 우선
#
# 의존: gemma_collate.py(parse_sample, load_audio) + gemma_collapse_check 라벨선택 로직.
#   faster-whisper(이미 설치됨). 라벨 선택은 본 파일에 자체 구현(Colab 의존 회피).
from __future__ import annotations

import argparse
import json
import random
from typing import Optional

from gemma_collate import load_audio, parse_sample


def is_pos(row: dict) -> bool:
    """metadata.label 기준 pos/neg(negative만 label='negative')."""
    label = row.get("metadata", {}).get("label")
    return label != "negative"


def select(jsonl: str, n: int, seed: int = 42):
    """pos/neg 각 n개 무작위 선택."""
    rows = [json.loads(l) for l in open(jsonl, "r", encoding="utf-8") if l.strip()]
    pos = [r for r in rows if is_pos(r)]
    neg = [r for r in rows if not is_pos(r)]
    random.seed(seed)
    random.shuffle(pos)
    random.shuffle(neg)
    return pos[:n], neg[:n]


def load_whisper(model_size: str):
    """faster-whisper 로드(로컬 GPU). 기존 dataset_transcriber 패턴."""
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cuda", compute_type="float16")


def transcribe(whisper, wav, sr: int) -> str:
    """오디오 배열 전사 -> 텍스트(빈 문자열이면 음성 없음). 세그먼트 합쳐 반환."""
    # faster-whisper는 16kHz float32 ndarray 입력 허용
    segments, _info = whisper.transcribe(wav, language="ko", beam_size=1,
                                         vad_filter=True)
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()


def scan(whisper, rows, base_dir: Optional[str], min_chars: int, tag: str):
    """rows 전사 -> 음성 클립(텍스트 min_chars 이상) 카운트. 샘플 출력."""
    voiced = 0
    samples = []
    for i, row in enumerate(rows):
        parsed = parse_sample(row)
        try:
            wav, sr = load_audio(parsed["audio_path"], base_dir=base_dir)
            text = transcribe(whisper, wav, sr)
        except Exception as e:                      # noqa: BLE001
            print(f"  {tag}[{i}] 전사 실패: {type(e).__name__}")
            continue
        if len(text) >= min_chars:
            voiced += 1
            if len(samples) < 3:
                samples.append(text[:60])
        if (i + 1) % 20 == 0:
            print(f"  {tag} 진행 {i + 1}/{len(rows)} (음성 {voiced})")
    return voiced, samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma 클립 음성 유무 Whisper 스캔")
    ap.add_argument("--jsonl", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--n", type=int, default=80, help="pos/neg 각 표본 수")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--model", default="medium", help="medium 또는 large-v3-turbo")
    ap.add_argument("--min-chars", type=int, default=10,
                    help="음성 클립 판정 최소 글자수(노이즈 제외)")
    args = ap.parse_args()

    pos_rows, neg_rows = select(args.jsonl, args.n)
    print(f"표본: pos {len(pos_rows)} / neg {len(neg_rows)}")
    print(f"=== Whisper 로드: {args.model} (로컬 GPU) ===")
    whisper = load_whisper(args.model)

    print("=== pos 스캔 ===")
    pv, ps = scan(whisper, pos_rows, args.base_dir, args.min_chars, "pos")
    print("=== neg 스캔 ===")
    nv, ns = scan(whisper, neg_rows, args.base_dir, args.min_chars, "neg")

    np_, nn = len(pos_rows), len(neg_rows)
    print("-" * 56)
    print(f"음성 포함 클립: pos {pv}/{np_} ({pv / np_:.1%})  "
          f"neg {nv}/{nn} ({nv / nn:.1%})")
    if ps:
        print("pos 음성 예시:", " | ".join(ps))
    if ns:
        print("neg 음성 예시:", " | ".join(ns))
    print("-" * 56)
    total = np_ + nn
    voiced = pv + nv
    rate = voiced / total if total else 0.0
    if rate >= 0.3:
        print(f"판정: 음성 클립 {rate:.1%} -> 충분. transcript 분리도 측정 진행 가능. "
              "다음: 음성 클립만 모아 프레임+오디오+transcript 분리도 측정.")
    elif rate <= 0.1:
        print(f"판정: 음성 클립 {rate:.1%} -> 거의 없음(순수 무음 게임플레이). "
              "transcript 구조적 불가. '무음은 59% 본질적 한계' 확정. 타깃의 한계.")
    else:
        print(f"판정: 음성 클립 {rate:.1%} -> 부분적. 측정은 가능하나 표본 작음. "
              "음성 클립 수 보고 측정 진행 여부 판단.")


if __name__ == "__main__":
    main()