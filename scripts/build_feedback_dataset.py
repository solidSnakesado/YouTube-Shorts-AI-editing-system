# 계층: 스크립트 (CLI 진입점)
# 역할: 피드백 루프 D - 사람 OK/NO 피드백을 회귀 학습 JSONL로 변환 (E 재학습 입력)
# 의존: app.core.config(settings.DATABASE_URL), relabel_regression(REGRESSION_INSTRUCTION 단일 출처),
#       shorts 테이블의 train_sample_json/feedback (33일차 A-1/A-2가 적재)
# 34일차 신규: 피드백 -> 학습데이터 변환. OK->0.9 / NO->0.1 회귀 라벨
#   - DB enum은 대문자 'OK'/'NO' 저장 (33일차 주의2) -> 대문자 매칭
#   - NO 중 사유가 selection 아닌 것(경계/편집)은 기본 제외 (선택 라벨 오염 방지, 33일차 주의2)
#   - 고정 0.9/0.1 대신 윈도우별 원점수 반영 -> 타깃 연속 분포 유지 (이진 붕괴 방지)
#   - train_sample_json의 frame_paths/prompt 그대로 사용 (학습-추론 프롬프트 일치)
#   - 출력 스키마는 dataset_regression_p1.jsonl과 호환 (E에서 cat 병합 가능)

"""피드백 -> 회귀 학습 JSONL 변환 - shorts 테이블의 OK/NO 피드백을 engagement_score 라벨로"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# scripts를 'python scripts/...'로 실행할 때 app 패키지 import용 (프로젝트 루트 추가)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.core.config import settings

# 34일차: REGRESSION_INSTRUCTION 단일 출처 재사용 (3중 정의 방지, 33일차 주의7)
#   실행 방식(python -m scripts.X / python scripts/X.py) 양쪽 호환
try:
    from relabel_regression import REGRESSION_INSTRUCTION
except ModuleNotFoundError:
    from scripts.relabel_regression import REGRESSION_INSTRUCTION

# 34일차 (B-2 상대보정): 피드백을 고정값(0.9/0.1) 대신 예측값(hook_score) +- @ 로 변환
#   OK -> 원점수 +@ / NO -> 원점수 -@. 윈도우별 원점수가 달라 타깃이 연속 -> 이진 붕괴 방지
NUDGE_DELTA = 0.15
_NEUTRAL_BASE = 0.5     # hook_score 결측 시 중립 기준값

# 34일차: NO 사유 중 '구간 선택은 옳았던' 케이스 - 0.1 라벨링 시 선택 신호 오염
_NONSELECTION_REASONS = {"boundary", "editing"}

# 34일차: E 실험 최소 임계값 (핸드오프 5절) - 미달이어도 변환은 진행 (경고만)
_MIN_SAMPLES = 100
_MIN_VIDEOS = 3
_MIN_CLASS_RATIO = 0.30

def _db_path_from_url(url: str) -> str:
    """settings.DATABASE_URL에서 SQLite 파일 경로 추출 (migrate 스크립트와 동일 규칙)"""
    
    prefix = url.split(":///")
    if len(prefix) != 2:
        raise SystemExit(f"SQLite URL 형식이 아님: {url}")
    return prefix[1]

def _fetch_feedback_rows(db_path: str) -> list[dict]:
    """shorts 테이블에서 피드백 + train_sample_json 보유 행만 조회 (raw SQL)"""
    
    if not Path(db_path).is_file():
        raise SystemExit(f"DB 파일 없음: {db_path}")
    
    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            "SELECT id, feedback, feedback_reason, is_exploration, "
            "train_sample_json, model_version, hook_score FROM shorts "
            "WHERE feedback IS NOT NULL AND train_sample_json IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()

def _parse_window(window: str) -> tuple[float, float] | None:
    """train_sample_json의 'S-E' 윈도우 문자열 -> (clip_start, clip_end)"""
    
    parts = window.split("-")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None

def _parse_duration(prompt: str) -> float:
    """저장된 프롬프트에서 '전체 길이: N초' 파싱 (학습 프롬프트 재구성용, 실패 시 0.0)"""
    
    m = re.search(r"전체 길이:\s*([\d.]+)초", prompt)
    return float(m.group(1)) if m else 0.0

def _video_stem_from_paths(frame_paths: list) -> str:
    """프레임 경로(data/feedback_frames/{stem}/w../f0.jpg)에서 영상 stem 추출 (다양성 집계용)"""
    
    for p in frame_paths:
        parts = Path(p).parts
        if "feedback_frames" in parts:
            i = parts.index("feedback_frames")
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"

def _label_score(label: str, reason, base_score, include_nonselection_no: bool) -> float | None:
    """34일차 (B-2 상대보정): 원래 예측값(base_score) +- @. OK는 +@, NO는 -@
    NO+경계/편집은 None(제외). label은 OK/NO 확정 상태
    고정 0.9/0.1 대신 윈도우별 원점수 반영 -> 연속 분포 보존(이진 붕괴 방지)."""
    
    if label == "NO" and (not include_nonselection_no) and reason in _NONSELECTION_REASONS:
        return None # 구간 선택은 옳았음 -> 라벨링하면 선택 신호 오염
    base = base_score if isinstance(base_score, (int, float)) else _NEUTRAL_BASE
    delta = NUDGE_DELTA if label == "OK" else -NUDGE_DELTA
    return round(max(0.0, min(1.0, base + delta)), 4)

def _build_record(row: dict, score: float) -> dict | None:
    """피드백 행 1건 -> 회귀 학습 레코드 (dataset_regression_p1 호환), 프레임/파싱 실패 시 None"""
    
    try:
        sample = json.loads(row["train_sample_json"])
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"train_sample_json 파싱 실패: {row['id']}")
        return None
    
    frame_paths = sample.get("frame_paths", [])
    valid = [p for p in frame_paths if Path(p).is_file()]
    if not valid:
        logger.warning(f"유효 프레임 0개 (보존 디렉토리 삭제?): {row['id']}")
        return None
    
    window = _parse_window(sample.get("window", ""))
    if window is None:
        logger.warning(f"window 파싱 실패: {row['id']}")
        return None
    clip_start, clip_end = window

    duration = _parse_duration(sample.get("prompt", ""))
    if duration <= 0:
        logger.debug(f"duration 파싱 실패(프롬프트) -> 0초: {row['id']}")

    # 34일차: metadata는 train_data_loader 생성기 분기(_build_meta_text) 형식과 일치 ->
    #   highlight_count 포함으로 회귀 추론 프롬프트(_build_phase2_prompt)와 동일 재구성 보장
    metadata = {
        "video_title": sample.get("title", "알 수 없음"),
        "duration_sec": duration,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "transcript": sample.get("transcript", ""),
        "highlight_count": 1,
        # 34일차: 출처/추적 메타 (학습 프롬프트엔 미사용, 분석/라운드 지표용)
        "source": "feedback",
        "feedback": row["feedback"],
        "feedback_reason": row["feedback_reason"],
        "is_exploration": bool(row["is_exploration"]),  # 33일차 주의3: NULL -> False
        "model_version": row["model_version"],
        "base_score": row["hook_score"],                # 34일차 (B-2): 보정 전 원래 예측값 (라운드 분석/검증용)
        "shorts_id": row["id"],
    }
    return {
        "images": valid,
        "instruction": REGRESSION_INSTRUCTION,
        "output": json.dumps({"engagement_score": score}, ensure_ascii=False),
        "metadata": metadata,
    }

def _report(records: list, skipped: dict) -> None:
    """클래스 균형/영상 다양성/E 임계값 점검 로그 (핸드오프 5절)"""
    
    labels = Counter(r["metadata"]["feedback"] for r in records)
    videos = {_video_stem_from_paths(r["images"]) for r in records}
    n = len(records)

    logger.info("-" * 60)
    logger.info(
        f"변환 완료: {n}개 | OK:{labels.get('OK', 0)} NO:{labels.get('NO', 0)} | 영상:{len(videos)}개"
    )
    logger.info(
        f"제외: 비선택NO {skipped['nonselection_no']} | "
        f"프레임/파싱 {skipped['no_frame']} | 알수없는라벨 {skipped['other']}"
    )

    # E 실험 임계값 점검 (미달이어도 변환은 유효 - 경고만)
    if n < _MIN_SAMPLES:
        logger.warning(f"샘플 {n} < {_MIN_SAMPLES} - E 재학습엔 부족 (지표/파이프라인 검증용으로만)")
    if len(videos) < _MIN_VIDEOS:
        logger.warning(f"영상 {len(videos)} < {_MIN_VIDEOS} - 단일 영상 과적합 위험 (다양성 부족)")
    if n > 0:
        minor = min(labels.get("OK", 0), labels.get("NO", 0)) / n
        if minor < _MIN_CLASS_RATIO:
            logger.warning(f"소수 클래스 {minor:.0%} < {_MIN_CLASS_RATIO:.0%} - 업샘플링/손실가중 필요")

def build(db_path: str, output_path: Path, include_nonselection_no: bool) -> None:
    """피드백 행 -> 회귀 JSONL 변환 메인 흐림"""
    
    rows = _fetch_feedback_rows(db_path)
    logger.info(f"피드백 행 로드: {len(rows)}개 (feedback + train_sample_json 보유)")

    records = []
    skipped = {"nonselection_no":0, "no_frame": 0, "other": 0}
    for row in rows:
        label = (row["feedback"] or "").upper()     # 33일차 주의2: 대문자 매칭
        if label not in ("OK", "NO"):
            logger.warning(f"알 수 없는 feedback 값 '{row['feedback']}': {row['id']}")
            skipped["other"] += 1
            continue
        score = _label_score(label, row["feedback_reason"], row["hook_score"], include_nonselection_no)
        if score is None:
            skipped["nonselection_no"] += 1
            continue
        rec = _build_record(row, score)
        if rec is None:
            skipped["no_frame"] += 1
            continue
        records.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _report(records, skipped)
    logger.info(f"출력: {output_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="피드백 -> 회귀 학습 JSONL 변환 (34일차, 피드백 루프 D)")
    p.add_argument("--output", default="data/finetune/dataset_feedback_r1.jsonl", help="출력 JSONL")
    p.add_argument(
        "--include-nonselection-no", action="store_true",
        help="NO 사유가 경계/편집인 것도 -@ 보정 포함 (기본: 제외, 선택 라벨 오염 방지)"
    )
    args = p.parse_args()

    db_path = _db_path_from_url(settings.DATABASE_URL)
    logger.info(f"대상 DB: {db_path}")
    build(db_path, Path(args.output), args.include_nonselection_no)

if __name__ == "__main__":
    main()