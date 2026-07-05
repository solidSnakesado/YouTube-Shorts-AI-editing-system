# 역할: Gemma 3N 멀티모달을 SFC용 collate_fn - all.jsonl 1행([프레임 N + 오디오 + 텍스트])을
#       프로세서 입력 배치로 변환. Colab(A100) 학습 루프의 data_collator로 주입.
# 41일차 신규: Unsloth Gemma3N audio-only collate를 [프레임+오디오] 멀티모달로 교체하는 골격
# 50일차 수정(1회): load_audio에 pad_to_sec(기본 30.0) 무음 패딩 추가 — 수정본 기준 L78~104.
#   비표준 길이(30s 미만) 클립의 오디오 특징↔토큰 수 불일치 방지 (학습·추론 공용 함수라 정합 자동 유지)
#   - 결정적 부분(경로추출/프레임샘플링)은 로컬 ast/pyflakes + 모의 시뮬로 검증 완료
#   - 미디어 로드(PIL/soundfile)는 지연 import (로컬 미설치 통과, Colab 설치본 사용)
#   - processor 실호출부(COLAB)는 Gemma 3N 프로세서 실동작 의존 -> Colab CPU 스모크로 확정
#
# 사용 흐름 (Colab):
#   1) CPU 런타임에서 processor 로드 -> smoke_test()로 멀티모달 수용 + 실토큰수 측정 (0 CU)
#   2) max_frames / max_length 확정
#   3) A100 전환 후 본 학습 (build_collate_fn(processor,...) 결과를 Trainer data_collator로)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------
# 1. content-block 파서 (결정적 - 로컬 검증 대상)
# --------------------------------------------------------------

def parse_sample(example: dict) -> dict:
    """all.jsonl 1행 -> {frame_paths, audio_path, instruction, target}.
    
    구조(확정): messages[0].content = [image x N, audio, text],
                messages[1].content = [text].
    """

    user_content = example["messages"][0]["content"]
    frame_paths = [b["image"] for b in user_content if b.get("type") == "image"]
    audio_blocks = [b["audio"] for b in user_content if b.get("type") == "audio"]
    text_blocks = [b["text"] for b in user_content if b.get("type") == "text"]
    target_blocks = example["messages"][1]["content"]

    if not frame_paths:
        raise ValueError("프레임 경로 없음 (image 블록 0개)")
    if not audio_blocks:
        raise ValueError("오디오 경로 없음 (audio 블록 0개)")
    if not text_blocks:
        raise ValueError("지시문 없음 (text 블록 0개)")
    
    return {
        "frame_paths": frame_paths,
        "audio_path": audio_blocks[0],
        "instruction": text_blocks[0],
        "target": target_blocks[0]["text"],
    }

def sample_frames(frame_paths: list[str], max_frames: int) -> list[str]:
    """프레임이 max_frames 초과 시 균등 샘플링(시간 분포 보존). 이하면 그대로ㅓ.

    30프레임 = 비전토큰 다수 -> max_length 초과 위험. 기본값 보수적 권장
    max_frames<=0 이면 샘플링 비활성(전체 사용)
    """

    n = len(frame_paths)
    if max_frames <= 0 or n <= max_frames:
        return frame_paths
    stem = (n - 1) / (max_frames - 1) if max_frames > 1 else 0
    idx = sorted({int(round(i * stem)) for i in range(max_frames)})
    return [frame_paths[i] for i in idx]

# --------------------------------------------------------------
# 2. 미디어 로더 (지연 import - 로컬 정적검증 통과, Colab 실행)
# --------------------------------------------------------------

def load_images(frame_paths: list[str], base_dir: Optional[str] = None):
    """프레임 경로 리스트 -> PIL.Image(RGB) 리스트. base_dir로 cwd 보정."""

    from PIL import Image # COLAB 설치본
    imgs = []
    for p in frame_paths:
        fp = Path(base_dir) / p if base_dir else Path(p)
        imgs.append(Image.open(fp).convert("RGB"))
    return imgs

def load_audio(audio_path: str, target_sr: int = 16000, base_dir: Optional[str] = None,
               pad_to_sec: Optional[float] = 30.0):
    """오디오 경로 -> (mono float32 array, sr). target_sr로 리샘플(librosa 가용 시).

    50일차: pad_to_sec 지정 시 그 길이까지 뒤쪽 무음(0) 패딩.
      비표준 길이(30s 미만) 클립에서 오디오 특징 프레임 수 ↔ 텍스트 오디오
      자리표시 토큰 수 불일치("Audio features and audio tokens do not match")를
      방지. 30s 이상은 그대로(자르지 않음 — 프로세서가 처리). None이면 패딩 비활성.
    """

    import numpy as np       # 50일차: 무음 패딩용
    import soundfile as sf  # COLAB 설치본
    fp = Path(base_dir) / audio_path if base_dir else Path(audio_path)
    wav, sr = sf.read(str(fp), dtype="float32")
    if wav.ndim > 1:        # 스테레오 -> 모노
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass            # 리샘플 불가 시 원본 SR 유지 (프로세서가 처리)
    if pad_to_sec is not None:                          # 50일차: 뒤쪽 무음 패딩
        need = int(round(pad_to_sec * sr))
        if wav.shape[0] < need:
            wav = np.concatenate([wav, np.zeros(need - wav.shape[0], dtype=wav.dtype)])
    return wav, sr

# --------------------------------------------------------------
# 3. collate_fn 팩토리 (processor 주입식)
# --------------------------------------------------------------

def _find_subseq(row_ids, sub_ids):
    """41일차: row_ids(1D tensor)에서 sub_ids(1D tensor) 첫 등장 시작 인덱스. 없으면 None.

    응답 구분자 토큰열(<|turn>model\\n)을 시퀀스에서 찾아 프롬프트 마스킹 경계로 사용.
    torch import 불필요 - 텐서 메서드(unfold/all/nonzero)만 사용.
    """
    n, m = int(row_ids.shape[0]), int(sub_ids.shape[0])
    if m == 0 or n < m:
        return None
    windows = row_ids.unfold(0, m, 1)           # [n-m+1, m]
    matches = (windows == sub_ids).all(dim=1)
    nz = matches.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return None
    return int(nz[0].item())


def build_collate_fn(
    processor: Any,
    max_frames: int = 12,
    max_length: int = 2048,
    base_dir: Optional[str] = None,
    target_sr: int = 16000,
    truncate: bool = False,
    response_template: str = "<|turn>model\n",   # 41일차: 응답-only 마스킹 구분자
):
    """Trainer data_collator로 쓸 collate_fn 생성
    
    Args:
        processor: Gemma 3N AutoProcessor (Colab 주입)
        max_frames: 샘플당 사용 최대 프레임 수(토큰 예산 - 스모크로 확정)
        max_length: 토큰 상환 (스모그 측정값 기반 상향 가능)
        base_dir: jsonl 상대경로 기준 디렉토리 (Colab tar 추출 위치, 보통 None=cwd)
        truncate: True면 max_length로 자름. 기본 False - 자르면 타겟(어시스턴트)이
                  잘려 학습이 깨질 수 있음, 초과 시 truncate 대신 max_frames 축소 권장
    """

    # 41일차: 응답 구분자 토큰열 사전 계산 (응답-only 마스킹용)
    #   <|turn>model\n -> Gemma4 기준 [105, 4368, 107] (하드코딩 대신 토크나이저에서 도출)
    resp_seq = processor.tokenizer.encode(response_template, add_special_tokens=False)

    def collate_fn(batch: list[dict]) -> dict:
        texts: list[str] = []
        images_per_ex: list[list] = []
        audio_per_ex: list = []

        for example in batch:
            parsed = parse_sample(example)
            frames = sample_frames(parsed["frame_paths"], max_frames)

            # 프로세서용 messages 재구성: 샘플링된 프레임 수로 image 플레이스홀더 재생성
            #   (텍스트 템플릿의 image 자리수 = 실제 전달 이미지 수 일치 필수)
            user_blocks: list[dict] = [{"type": "image"} for _ in frames]
            user_blocks.append({"type": "audio"})
            user_blocks.append({"type": "text", "text": parsed["instruction"]})
            messages = [
                {"role": "user", "content": user_blocks},
                {"role": "assistant",
                 "content": [{"type": "text", "text": parsed["target"]}]}
            ]

            # COLAB: apply_char_template - Gemma 3N 템플릿이 assistant->'<start_of_turn>model' 렝더
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images_per_ex.append(load_images(frames, base_dir))
            wav, _sr = load_audio(parsed["audio_path"], target_sr, base_dir)
            audio_per_ex.append(wav)

        # COLAB -결정적 검증 지점 - Gemma 3N 프로세서가 text+images_audio 동시 수용?
        #   아래는 HF 멀티모달 표준 시그니쳐(예상). 스모크에서 키워드/형태 확정:
        #   - images 인자: 배치별 리스트의 리스트 vs 평탄화 - 프로세서마다 다름
        #   - audio 인자명: 'audio'(Unsloth Gemma3N 노트북 기준), 일부는 'audios'
        proc_kwargs: dict = dict(
            text=texts, images=images_per_ex, audio=audio_per_ex,
            return_tensors="pt", padding=True,
        )
        if truncate:
            proc_kwargs["truncation"] = True
            proc_kwargs["max_length"] = max_length
        inputs = processor(**proc_kwargs)

        # 41일차: labels - 응답-only 마스킹 (어시스턴트 응답만 학습)
        #   1) 패딩 -100  2) 응답 구분자 끝까지 -100 -> 프롬프트/이미지/오디오/지시문 제외
        import torch
        labels = inputs["input_ids"].clone()
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        resp = torch.tensor(resp_seq, device=labels.device)
        m = int(resp.shape[0])
        for i in range(int(labels.shape[0])):
            start = _find_subseq(inputs["input_ids"][i], resp)
            if start is None:
                labels[i, :] = -100               # 구분자 없으면 전체 마스킹(프롬프트 학습 방지)
            else:
                labels[i, : start + m] = -100     # 구분자(끝)까지 마스킹 -> 응답만 학습
        inputs["labels"] = labels
        return inputs
    return collate_fn

# --------------------------------------------------------------
# 4. 스모크 테스트 (Colab CPU에서 먼저 - A100 불필요, 0 CU)
# --------------------------------------------------------------

def smoke_test(
    jsonl_path: str,
    processor: Any,
    max_frames: int = 12,
    max_length: int = 2048,
    base_dir: Optional[str] = None,
    n_samples: int = 2,
) -> dict:
    """프로세서에 1~2샘플 통과 -> 멀티모달 수용 + 실토큰수 측정
    
    통과: 결합 모달리티 수용 실증 + 토큰 예산 확정 -> 본 학습 진행
    실패: audio-only / frames-only / 프레임수 축소로 재설계 분기
    * truncate=False(기본)로 실토큰수를 그대로 측정 - over_max_length로 초과 판정
    """

    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n_samples:
                break

    collate = build_collate_fn(
        processor, max_frames=max_frames, max_length=max_length,
        base_dir=base_dir, truncate=False)
    inputs = collate(rows)
    seq_len = int(inputs["input_ids"].shape[1])
    report = {
        "samples": len(rows),
        "seq_len": seq_len,
        "max_length": max_length,
        "over_max_length": seq_len >= max_length,
        "keys": sorted(inputs.keys()),
        "input_ids_shape": tuple(inputs["input_ids"].shape),
    }
    print("=== Gemma 멀티모달 스모크 ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if report["over_max_length"]:
        print(f"  ⚠️ seq_len {seq_len} >= max_length {max_length}"
              f" -> max_frames 축소 또는 max_length 상향 필요")
    else:
        print(f"  ✓ 토큰 예산 OK (여유 {max_length - seq_len})")
    return report