# 계층: 스크립트 (CLI 진입점)
# 역할: train_qlora.py 데이터 로드/포맷 변환 함수 분리 (300줄 원칙 대응)
# 27일차 신규: dataset.jsonl -> QLoRA 학습 -> LoRA 어댑터 저장

"""QLoRA 학습용 데이터 로드 및 대화 포맷 변환"""

import json
from pathlib import Path

from loguru import logger

def load_dataset_jsonl(jsonl_path: Path) -> list[dict]:
    """
    dataset.jsonl 로드 + 이미지 파일 존재 검증

    Args:
        jsonl_path: 데이터셋 JSONL 파일 경로

    Returns:
        유효한 샘플 리스트 (이미지 존재 확인 완료)
    """

    if not jsonl_path.is_file():
        raise FileExistsError(f"데이터셋 없음: {jsonl_path}")
    
    samples = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"JSON 파싱 실패 (라인 {line_num})")
                skipped += 1
                continue

            # 이미지 파일 존재 확인 (최소 1장)
            images = record.get("images", [])
            valid_images = [p for p in images if Path(p).is_file()]
            if not valid_images:
                skipped += 1
                continue

            record["images"] = valid_images
            samples.append(record)

    logger.info(f"데이터 로드: {len(samples)}개 샘플, {skipped}개 스킵")
    return samples

def build_conversation_format(samples: list[dict]) -> list[dict]:
    """
    Unsloth SFT 학습용 대화 포맷으로 변환
    각 샘플 -> messages 리스트 (user: 이미지 + 텍스트, assistant: 라벨)

    이미지 최대 3장 제한 - VRAM 절약 (비주얼 토큰 ~840개)
    """

    conversations = []
    for sample in samples:
        images = sample.get("images", [])
        instruction = sample.get("instruction", "")
        metadata = sample.get("metadata", {})
        label = sample.get("output", "일반")

        user_content = []
        for img_path in images[:3]:
            user_content.append({"type": "image", "image": str(Path(img_path).resolve())})

        meta_text = (
            f"영상: {metadata.get('video_title', '알 수 없음')}\n"
            f"구간: {metadata.get('segment_start', 0):.1f}초 ~ "
            f"{metadata.get('segment_end', 0):.1f}초\n"
            f"전체 길이: {metadata.get('duration_sec', 0):.0f}초\n"
            f"위치: {metadata.get('position_ratio', 0):.1%}\n\n"
            f"{instruction}"
        )
        user_content.append({"type": "text", "text": meta_text})

        conversations.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label},
            ]
        })

    logger.info(f"대화 포맷 변환 완료: {len(conversations)}개")
    return conversations