# 계층: 스크립트 (CLI 진입점)
# 역할: 게이트 진단 (1) 채점 - 사람 맹검 라벨과 히트맵 정답을 비교하여 정확도 산출
# 의존: gate_test1_prepare.py 출력 (label_blind.csv, answers.json)
# 33일차 신규: 입력 충분성 게이트 판정
#   - 정확도 ≈ 50% → 5프레임에 신호 없음 (입력 부족 확정)
#   - 정확도 ≳ 75% → 신호 존재 (피드백 루프 유효)

"""게이트 진단 (1) 채점 - 맹검 라벨 정확도 + 혼동 행렬 + 판정"""

import argparse
import csv
import json
from pathlib import Path

from loguru import logger


def load_blind_labels(path: Path) -> dict:
    """label_blind.csv에서 사람 라벨 읽기 -> {window_id: label}"""

    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row["window_id"].strip()
            label = row["your_label(H/N)"].strip().upper()
            if label in ("H", "N"):
                labels[wid] = label
            else:
                logger.warning(f"{wid}: 유효하지 않은 라벨 '{label}' (H/N만 허용), 건너뜀")
    return labels


def load_answers(path: Path) -> dict:
    """answers.json에서 정답 읽기"""

    return json.loads(path.read_text(encoding="utf-8"))


def score(blind: dict, answers: dict) -> None:
    """정확도, 혼동 행렬, 게이트 판정 출력"""

    # 혼동 행렬 카운터
    tp = fp = fn = tn = 0
    details = []

    for wid in sorted(blind.keys()):
        if wid not in answers:
            logger.warning(f"{wid}: answers.json에 없음, 건너뜀")
            continue
        human = blind[wid]
        true = answers[wid]["true_label"]
        eng = answers[wid]["engagement"]
        hit = "✅" if human == true else "❌"

        if human == "H" and true == "H":
            tp += 1
        elif human == "H" and true == "N":
            fp += 1
        elif human == "N" and true == "H":
            fn += 1
        else:
            tn += 1

        details.append(f"  {wid} | 사람:{human} 정답:{true} eng:{eng:.2f} {hit}")

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # 상세 결과
    logger.info("=" * 60)
    logger.info("게이트 진단 (1) 채점 결과")
    logger.info("=" * 60)
    for line in details:
        logger.info(line)

    # 혼동 행렬
    logger.info("-" * 60)
    logger.info("혼동 행렬:")
    logger.info(f"                정답 H    정답 N")
    logger.info(f"  사람 H          {tp:3d}       {fp:3d}    (정밀도: {precision:.1%})")
    logger.info(f"  사람 N          {fn:3d}       {tn:3d}")
    logger.info(f"                (재현율: {recall:.1%})")

    # 요약
    logger.info("-" * 60)
    logger.info(f"전체 정확도: {accuracy:.1%} ({tp + tn}/{total})")
    logger.info(f"정밀도(H): {precision:.1%} | 재현율(H): {recall:.1%}")

    # 게이트 판정
    logger.info("=" * 60)
    if accuracy <= 0.55:
        logger.info("🔴 판정: 5프레임에 하이라이트 신호 없음 (우연 수준)")
        logger.info("   → 입력 부족 확정. 피드백 루프 전에 입력 보강 필요 (OCR, 오디오, 더 긴 클립)")
    elif accuracy <= 0.70:
        logger.info("🟡 판정: 약한 신호 존재 (부분적 판별 가능)")
        logger.info("   → 피드백 루프 가능하나 천장 낮음. 입력 보강 병행 권장")
    else:
        logger.info("🟢 판정: 신호 충분 (사람이 5프레임으로 판별 가능)")
        logger.info("   → 피드백 루프로 모델 개선 유효. 데이터 양이 병목")
    logger.info("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="게이트 진단(1) 채점 (33일차)")
    p.add_argument("--dir", default="data/gate_test1", help="gate_test1_prepare.py 출력 디렉토리")
    args = p.parse_args()

    d = Path(args.dir)
    blind = load_blind_labels(d / "label_blind.csv")
    answers = load_answers(d / "answers.json")

    if not blind:
        raise SystemExit("label_blind.csv에 유효한 라벨 없음")

    logger.info(f"라벨 로드: 사람 {len(blind)}개 / 정답 {len(answers)}개")
    score(blind, answers)


if __name__ == "__main__":
    main()