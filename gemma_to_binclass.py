# 45일차 수정(3회) | 배치: ~/project/yt_shorts_ai/gemma_to_binclass.py
#
# [수정 1회] 라벨 기준 변경: hook_score 임계 -> metadata.label.
# [수정 2회] 라벨 판정 규칙 정정: 데이터에 positive는 label 키 없음(negative만 보유)
#   -> "negative"만 neg, 그 외(None 포함)는 pos.
# [수정 3회] instruction(프롬프트) 동시 교체 추가: 기존 지시문이 회귀/JSON용
#   ("hook_score JSON 또는 빈 리스트")이라 분류 target("상"/"하")과 불일치 -> 학습신호 오염.
#   target 교체와 함께 instruction을 분류용("'상'/'하'로만 답하세요")으로 일괄 교체.
#   --instruction 템플릿({pos}/{neg} 치환), 빈 문자열이면 교체 안 함.
#   변경 라인(이 파일 기준): 아래 전달 메시지 참조.
#
# 목적: 분류 버킷 전환(방향1, 2클래스) 1단계. 단일점수 회귀 5번 붕괴 ->
#   회귀("전부 0/평균값" escape) 대신 이산 등급 토큰으로 escape 비용 증가.
#   분리도 진단(학습0): 임계 0.5 정확도 상한 94.8%. metadata 기준 pos1752/neg1595
#   (점수임계 기준이면 상1578/하1769, 차이 174=경계 노이즈).
#
#   변환: metadata.label=="negative" -> neg 라벨 / 그 외 -> pos 라벨 (+ instruction 교체)
#         (재라벨 완료본 train_relabel/eval_relabel 대상 - target/instruction만 교체)
#
#   32일차 교훈(이진분류 붕괴=출력 길이 차이) 회피: pos/neg 라벨은 길이 동일 토큰 권장.
#   토크나이저 단일토큰 여부는 학습 셀에서 확인 권장. 라벨/임계/지시문은 인자로 변경 가능.
#   collate는 target을 문자열로만 다루므로 수정 불필요. 붕괴 체크 파서는 별도 변경.
#
# 의존: 없음(표준 라이브러리만).
# 실행(로컬 WSL):
#   python gemma_to_binclass.py
#   (기본: relabel -> binclass, metadata.label 기준, pos=상/neg=하, 분류 지시문 교체)
from __future__ import annotations

import argparse
import json
import re

# 분류용 지시문 템플릿({pos}/{neg}는 실제 라벨 토큰으로 치환). 회귀/JSON 지시문 대체.
DEFAULT_INSTRUCTION = (
    "영상 프레임과 오디오를 분석하여 이 클립이 쇼츠 하이라이트인지 판단하세요. "
    "하이라이트면 '{pos}', 아니면 '{neg}'로만 답하세요."
)


def extract_score(target_text: str) -> float | None:
    """target(JSON 또는 순수숫자) -> hook_score float. 빈 리스트 0.0. 실패 None.
    (라벨 결정엔 미사용 - metadata와 점수의 불일치 진단용)."""
    try:
        obj = json.loads(target_text)
    except json.JSONDecodeError:
        # JSON 아니면 숫자 직접 시도(이미 숫자 변환된 경우)
        m = re.fullmatch(r"\s*([01]?\.\d+|[01])\s*", target_text)
        return float(m.group(1)) if m else None
    if isinstance(obj, (int, float)):
        return float(obj)                           # json이 순수숫자를 파싱한 경우
    if not isinstance(obj, dict):
        return None
    hl = obj.get("highlights")
    if not isinstance(hl, list):
        return None
    if len(hl) == 0:
        return 0.0                                  # 빈 리스트 -> 0.0
    first = hl[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return float(first["hook_score"])
        except (TypeError, ValueError):
            return None
    return None


def replace_instruction(row: dict, new_instr: str) -> bool:
    """messages[0]의 첫 text 블록 text를 new_instr로 교체. 성공 시 True."""
    try:
        blocks = row["messages"][0]["content"]
    except (KeyError, IndexError, TypeError):
        return False
    for block in blocks:
        if block.get("type") == "text":
            block["text"] = new_instr
            return True
    return False


def convert_file(in_path: str, out_path: str, thr: float,
                 pos_lbl: str, neg_lbl: str, instr: str) -> dict:
    """target을 metadata.label 기준 2클래스 라벨로 변환 + instruction 교체 -> 새 파일.
    점수 임계와 metadata가 어긋난 경계 클립 수도 보고(라벨 결정엔 미사용)."""
    pos = neg = fail = mismatch = replaced = 0
    out_rows: list[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("metadata", {}).get("label")
            # 데이터 구조: negative만 label="negative", positive는 label 키 없음.
            # 진단/gemma_label_dist.py와 동일 -> "negative"만 neg, 그 외(None 포함)는 pos.
            is_pos = label != "negative"
            try:
                tgt = row["messages"][1]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                fail += 1
                continue
            row["messages"][1]["content"][0]["text"] = pos_lbl if is_pos else neg_lbl
            if instr and replace_instruction(row, instr):
                replaced += 1
            # 점수 임계와 metadata 불일치 진단(경계 노이즈, 라벨 결정엔 미사용)
            score = extract_score(tgt)
            if score is not None and (score >= thr) != is_pos:
                mismatch += 1
            out_rows.append(row)
            if is_pos:
                pos += 1
            else:
                neg += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== {in_path} -> {out_path} ===")
    print(f"pos({pos_lbl}) {pos} | neg({neg_lbl}) {neg} | 실패 {fail} "
          f"| 출력 {len(out_rows)}행")
    print(f"  instruction 교체: {replaced}행")
    print(f"  경계 노이즈(점수임계 {thr}와 metadata 불일치): {mismatch}")
    if out_rows:
        print(" 변환 샘플(target):")
        for row in out_rows[:4]:
            print(f"   {row['messages'][1]['content'][0]['text']}")
    return {"pos": pos, "neg": neg, "fail": fail,
            "mismatch": mismatch, "replaced": replaced}


def main() -> None:
    ap = argparse.ArgumentParser(description="metadata.label -> 2클래스 라벨 + 지시문 교체")
    ap.add_argument("--in-train", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--in-eval", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--out-train", default="datasets/gemma_audio/train_binclass.jsonl")
    ap.add_argument("--out-eval", default="datasets/gemma_audio/eval_binclass.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pos-label", default="상")
    ap.add_argument("--neg-label", default="하")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                    help="분류 지시문 템플릿({pos}/{neg} 치환). 빈 문자열이면 교체 안 함")
    args = ap.parse_args()

    instr = args.instruction.replace("{pos}", args.pos_label).replace("{neg}", args.neg_label)
    convert_file(args.in_train, args.out_train, args.threshold,
                 args.pos_label, args.neg_label, instr)
    convert_file(args.in_eval, args.out_eval, args.threshold,
                 args.pos_label, args.neg_label, instr)
    print("\n완료(metadata 기준 + 지시문 교체). 붕괴 체크 파서(등급) + collate 확인 후 재번들.")


if __name__ == "__main__":
    main()