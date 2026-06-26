# 44일차 신규 | 전체 신규 | 배치: ~/project/yt_shorts_ai/gemma_to_numeric.py
#
# 목적: (가) 출력 형식 재설계 1단계. 붕괴 원인=고정 JSON 토큰
#   ({"highlights": [{"hook_score": , }]})이 cross-entropy 지배 -> 점수 신호 희석.
#   target을 순수 숫자 문자열로 변환해 고정 토큰을 원천 제거.
#
#   변환: {"highlights": [{"hook_score": 0.73}]} -> "0.73"
#         {"highlights": []}                      -> "0.0" (혹시 남은 빈 라벨)
#   (이미 재라벨된 train_relabel/eval_relabel 대상 - 데이터 재생성 불필요, target만 교체)
#
#   user/instruction/metadata/frames/audio는 그대로. assistant target 텍스트만 변경.
#   collate는 target을 문자열로만 다루므로(형식 무관) 수정 불필요. 파서는 별도 변경.
#
# 의존: 없음(표준 라이브러리만). 입력 jsonl의 messages[1].content[0].text만 파싱.
# 실행(로컬 WSL):
#   python gemma_to_numeric.py
#   (기본: train_relabel/eval_relabel -> train_numeric/eval_numeric)
from __future__ import annotations

import argparse
import json
import re


def extract_score(target_text: str) -> float | None:
    """target JSON 문자열 -> hook_score float. 빈 리스트는 0.0. 실패는 None."""
    try:
        obj = json.loads(target_text)
    except json.JSONDecodeError:
        # JSON 아니면 숫자 직접 시도(이미 변환된 경우)
        m = re.fullmatch(r"\s*([01]?\.\d+|[01])\s*", target_text)
        return float(m.group(1)) if m else None
    hl = obj.get("highlights")
    if not isinstance(hl, list):
        return None
    if len(hl) == 0:
        return 0.0                              # 빈 리스트 -> 0.0
    first = hl[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return float(first["hook_score"])
        except (TypeError, ValueError):
            return None
    return None


def convert_file(in_path: str, out_path: str) -> dict:
    """jsonl의 assistant target을 숫자 문자열로 변환 -> 새 파일. 통계 반환."""
    ok = fail = 0
    out_rows: list[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                tgt = row["messages"][1]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                fail += 1
                continue
            score = extract_score(tgt)
            if score is None:
                fail += 1
                continue
            # 숫자 문자열로 교체(소수 4자리, 불필요한 0 정리는 안 함 - 일관 형식)
            row["messages"][1]["content"][0]["text"] = f"{round(score, 4)}"
            out_rows.append(row)
            ok += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== {in_path} -> {out_path} ===")
    print(f"변환 {ok} | 실패 {fail} | 출력 {len(out_rows)}행")
    # 변환 샘플 3개
    print(" 변환 샘플(target):")
    for row in out_rows[:3]:
        print(f"   {row['messages'][1]['content'][0]['text']}")
    return {"ok": ok, "fail": fail}


def main() -> None:
    ap = argparse.ArgumentParser(description="(가) target JSON -> 순수 숫자 변환")
    ap.add_argument("--in-train", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--in-eval", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--out-train", default="datasets/gemma_audio/train_numeric.jsonl")
    ap.add_argument("--out-eval", default="datasets/gemma_audio/eval_numeric.jsonl")
    args = ap.parse_args()

    convert_file(args.in_train, args.out_train)
    convert_file(args.in_eval, args.out_eval)
    print("\n완료. 파서 변경 + collate 확인 후 재번들 -> 재학습.")


if __name__ == "__main__":
    main()