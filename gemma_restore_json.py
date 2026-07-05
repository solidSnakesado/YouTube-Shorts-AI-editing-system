# 46일차 수정(1회) | 배치: ~/project/yt_shorts_ai/gemma_restore_json.py
# [수정 1회] extract_score: json.loads가 숫자('1.0'->float)를 반환하는 경우 처리 추가
#   (L29 부근). 사유: numeric 타깃 '1.0'은 JSONDecodeError 안 나고 float로 파싱돼
#   dict 분기서 전량 None 실패 -> 숫자 반환 분기 추가.
#
# 목적: 붕괴 원인 검증. Qwen(성공)과 Gemma(0.111 붕괴)의 유일한 차이 = 출력 형식.
#   Qwen: {"engagement_score": 0.73} (JSON 래퍼) -> 붕괴 없음
#   Gemma round8: 0.73 (순수 숫자, round8서 JSON 벗김) -> 0.111 붕괴
#   라벨 생성 로직/값은 양쪽 동일(클립 구간 히트맵 가중평균 graded) -> JSON 래퍼만 차이.
#   가설: 순수 스칼라는 상수(0.111)로 도망치기 쉽고, JSON 구조가 앵커 역할로 붕괴 방지.
#   -> Gemma 타깃을 Qwen과 동일 형식으로 바꿔 재학습 -> 붕괴 사라지면 JSON 제거가 원인 확정.
#
# 변환: train_relabel/eval_relabel.jsonl(graded, {"highlights":[{"hook_score":X}]})에서
#   hook_score 추출 -> {"engagement_score": round(X,2)} 로 재포장(Qwen relabel_regression
#   L104/L108과 동일: round 2자리 + engagement_score 키). 라벨 값 불변, 형식만 교체.
#   numeric이 아니라 relabel을 입력으로 쓰는 이유: numeric은 이미 round4자리 손실, relabel이
#   원본 graded 값 보존. 빈 라벨({"highlights":[]})은 engagement_score 0.0.
#
# 입력: train_relabel.jsonl / eval_relabel.jsonl (44일차 graded 재라벨 산출물)
# 출력: train_qwenfmt.jsonl / eval_qwenfmt.jsonl (타깃만 Qwen 형식, 나머지 동일)
# 의존: 표준 라이브러리만. messages[1].content[0].text만 교체(collate는 형식 무관).
# 실행: python gemma_to_qwen_format.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_NUM_RE = re.compile(r"\s*([01]?\.\d+|[01])\s*")


def extract_score(target_text: str):
    """target -> hook_score float. {"highlights":[{"hook_score":X}]} 또는 빈/숫자 대응."""
    try:
        obj = json.loads(target_text)
    except json.JSONDecodeError:
        m = _NUM_RE.fullmatch(target_text)          # 이미 순수 숫자 문자열인 경우
        return float(m.group(1)) if m else None
    if isinstance(obj, (int, float)):               # '1.0'/'0.73' -> json.loads가 숫자 반환
        return float(obj)
    if isinstance(obj, dict) and "engagement_score" in obj:
        try:                                        # 이미 qwen 형식이면 그대로
            return float(obj["engagement_score"])
        except (TypeError, ValueError):
            return None
    hl = obj.get("highlights") if isinstance(obj, dict) else None
    if hl is None:
        return None
    if len(hl) == 0:
        return 0.0
    first = hl[0]
    if isinstance(first, dict) and "hook_score" in first:
        try:
            return float(first["hook_score"])
        except (TypeError, ValueError):
            return None
    return None


def convert(in_path: Path, out_path: Path) -> dict:
    """assistant target을 {"engagement_score": round(X,2)}로 변환. 통계 반환."""
    stats = {"total": 0, "converted": 0, "failed": 0}
    bins = [0] * 10
    rows_out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            row = json.loads(line)
            try:
                tgt = row["messages"][1]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                stats["failed"] += 1
                continue
            score = extract_score(str(tgt))
            if score is None:
                stats["failed"] += 1
                continue
            score = round(max(0.0, min(1.0, score)), 2)     # Qwen과 동일 round 2자리
            bins[min(int(score * 10), 9)] += 1
            row["messages"][1]["content"][0]["text"] = json.dumps(
                {"engagement_score": score}, ensure_ascii=False)
            rows_out.append(row)
            stats["converted"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows_out:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {in_path.name} -> {out_path.name}: "
          f"변환 {stats['converted']}/{stats['total']} (실패 {stats['failed']})")
    print("    점수 분포(0.1단위): "
          + " ".join(f"{i / 10:.1f}:{bins[i]}" for i in range(10)))
    if rows_out:
        print(f"    샘플 target: {rows_out[0]['messages'][1]['content'][0]['text']}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma target -> Qwen {engagement_score} 형식")
    ap.add_argument("--in-train", default="datasets/gemma_audio/train_relabel.jsonl")
    ap.add_argument("--in-eval", default="datasets/gemma_audio/eval_relabel.jsonl")
    ap.add_argument("--out-train", default="datasets/gemma_audio/train_qwenfmt.jsonl")
    ap.add_argument("--out-eval", default="datasets/gemma_audio/eval_qwenfmt.jsonl")
    args = ap.parse_args()

    print("=== Gemma target -> Qwen 형식 변환 ({\"engagement_score\": X}) ===")
    convert(Path(args.in_train), Path(args.out_train))
    convert(Path(args.in_eval), Path(args.out_eval))
    print("\n완료. 다음: 이 데이터로 gemma_retrain 재학습 -> 0.111 붕괴 사라지는지 확인.")
    print("  (라벨값/입력/모델 동일, 출력 형식만 Qwen 스타일 -> JSON 래퍼가 붕괴 막는지 시험)")


if __name__ == "__main__":
    main()