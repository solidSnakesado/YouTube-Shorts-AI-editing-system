# 계층: 스크립트 (CLI 진입점)
# 역할: train_qlora.py 데이터 로드/포맷 변환 함수 분리 (300줄 원칙 대응)
# 27일차 신규: dataset.jsonl -> QLoRA 학습 -> LoRA 어댑터 저장
# 31일차: Phase 2 - 클립 구간 + Whisper 전사 텍스트 프롬프트 반영

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

def _build_meta_text(metadata: dict, instruction: str) -> str:
    """판별기/생성기/Phase 2 메타데이터 형식에 따라 프롬프트 텍스트 생성"""

    title = metadata.get("video_title", "알 수 없음")
    duration = metadata.get("duration_sec", 0)

    # Phase 2 생성기: clip_start 존재 (10초 클립 + Whisper 전사 텍스트)
    if "clip_start" in metadata:
        text = (
            f"영상: {title}\n"
            f"클립: {metadata['clip_start']:.1f}초 ~ {metadata['clip_end']:.1f}초\n"
            f"전체 길이: {duration:.0f}초\n"
        )
        transcript = metadata.get("transcript", "")
        if transcript:
            text += f"전사: {transcript}\n"
        text += f"\n{instruction}"
        return text
    # 생성기: highlight_count 존재, segment_start 없음
    if "highlight_count" in metadata:
        return (
            f"영상: {title}\n"
            f"전체 길이: {duration:.0f}초\n\n"
            f"{instruction}"
        )
    # 판별기: segment_start/segment_end 존재
    return (
        f"영상: {title}\n"
        f"구간: {metadata.get('segment_start', 0):.1f}초 ~ "
        f"{metadata.get('segment_end', 0):.1f}초\n"
        f"전체 길이: {duration:.0f}초\n"
        f"위치: {metadata.get('position_ratio', 0):.1%}\n\n"
        f"{instruction}"
    )

def build_conversation_format(samples: list[dict]) -> list[dict]:
    """
    Unsloth SFT 학습용 대화 포맷으로 변환
    각 샘플 -> messages 리스트 (user: 이미지 + 텍스트, assistant: 라벨)

    판별기: 이미지 최대 3장 제한 - VRAM 절약 (비주얼 토큰 ~840개)
    생성기: 이미지 최대 5장 (다수 피크 커버)
    Phase 2 클립: 이미지 최대 10장 (1fps x 10초, 336px -> ~1,200토큰)
    """

    conversations = []
    for sample in samples:
        images = sample.get("images", [])
        instruction = sample.get("instruction", "")
        metadata = sample.get("metadata", {})
        label = sample.get("output", "일반")

        is_p2_clip = "clip_start" in metadata
        is_generator = "highlight_count" in metadata
        max_images = 10 if is_p2_clip else (5 if is_generator else 3)

        user_content = []
        for img_path in images[:max_images]:
            user_content.append({"type": "image", "image": str(Path(img_path).resolve())})

        meta_text = _build_meta_text(metadata, instruction)
        user_content.append({"type": "text", "text": meta_text})

        conversations.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label},
            ]
        })

    logger.info(f"대화 포맷 변환 완료: {len(conversations)}개")
    return conversations