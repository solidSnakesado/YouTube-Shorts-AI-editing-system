# 계층: 비즈니스 로직 계층 (Service 헬퍼) | 의존: config, llm_server, frame_extractor
# 역할: VLM 멀티모달 분석 오케스트레이션 (llama-server / 생성기 LoRA / 판별기 LoRA)
# 23일차: 생성기 -> 판별기 LoRA 순차 파이프라인 추가
# 31일차: Phase 2 슬라이딩 윈도우 추론으로 전환 (학습-추론 프롬프트 일치)

"""VLM 클라이언트 - 영상 프레임 + 텍스트 통합 분석 (Qwen2.5-VL-7B + LoRA)"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from loguru import logger

from app.core.config import settings
from app.core.llm_server import start_llm_server, stop_llm_server
from app.services.frame_extractor import extract_frames
from app.services.lora_utils import load_lora_model, unload_lora_model, frames_to_pil, lora_generate

# parse_highlights 재사용: VLM 응답도 동일한 JSON 구조이므로 기존 파서 활용
from app.services.llm_highlight_extractor import parse_highlights, MIN_DURATION_SEC, MAX_DURATION_SEC

def is_vlm_available() -> bool:
    """VLM 사용 가능 여부 (llama-server + GGUF + mmproj 존재 확인)
    33일차: LoRA 경로는 Unsloth 직접 추론이라 llama-server/mmproj 불필요 -> 단락 처리"""

    if settings.LORA_ENABLED and (settings.lora_generator_path.exists() or settings.lora_phase1_path.exists()):
        return True

    server_ok = Path(settings.LLAMA_SERVER_PATH).is_file()
    mmproj_ok = settings.mmproj_model_file.is_file()

    if not server_ok:
        logger.debug("VLM 불가: llama-server 바이너리 미존재")
    if not mmproj_ok:
        logger.debug("VLM 불가: mmproj 파일 미존재")

    return server_ok and mmproj_ok

def _is_server_running() -> bool:
    """llama-server가 이미 실행 중인지 /health 엔드포인트로 확인"""

    url = f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}/health"
    try:
        resp = urlopen(url, timeout=2)
        return resp.status == 200
    except (URLError, OSError):
        return False

async def run_vlm_analysis(source_path: str, transcript_data: dict, max_shorts: int = 5, lora_adapter_path: Optional[str] = None) -> list[dict]:
    """VLM 멀티모달 분석 파이프라인 실행
    LORA_ENABLED=true: 생성기 LoRA -> 판별기 LoRA 순차 실행
    lora_adapter_path 직접 지정: 해당 어댑터 단독 실행 (evaluate_lora 등)
    기본: llama-server 서브프로세스"""

    total_duration = transcript_data.get("duration_sec", 0)
    target_dur = int(transcript_data.get("target_duration_sec") or 0)
    loop = asyncio.get_event_loop()

    # 직접 어댑터 지정 (evaluate_lora.py 등 외부 호출)
    if lora_adapter_path and Path(lora_adapter_path).exists():
        logger.info(f"LoRA 단독 추론: {lora_adapter_path}")
        frames = await _extract_default_frames(source_path, total_duration)
        text = await loop.run_in_executor(
            None, _run_lora_inference, frames, transcript_data, max_shorts, lora_adapter_path)
        return parse_highlights(text, total_duration, max_shorts, target_duration_sec=target_dur)
    
    # 31일차: Phase 2 슬라이딩 윈도우 추론 (프레임 추출은 phase2_inference 내부에서 처리)
    # 33일차: LORA_PIPELINE=phase1 시 Phase 1 생성기 단독 1회 추론 (품질 비교 테스트, 판별기 제외)
    gen_path = settings.lora_generator_path
    if settings.LORA_ENABLED and settings.LORA_PIPELINE == "phase1":
        p1_path = settings.lora_phase1_path
        if not p1_path.exists():
            logger.error(f"Phase 1 어댑터 없음: {p1_path}")
            return []
        logger.info(f"Phase 1 생성기 추론 (품질 비교 테스트): {p1_path}")
        frames = await _extract_default_frames(source_path, total_duration)
        text = await loop.run_in_executor(
            None, _run_lora_inference, frames, transcript_data, max_shorts, str(p1_path))
        candidates = parse_highlights(text, total_duration, max_shorts, target_duration_sec=target_dur)

        # 33일차 (테스트 b): 판별기 검증 - 서브프로세스 VRAM 분리, 실패 시 후보 그대로 통과
        version_tag = str(p1_path)
        if settings.LORA_PHASE1_VERIFY and settings.lora_adapter_path.exists():
            logger.info(f"Phase 1 판별기 검증: {settings.lora_adapter_path}")
            candidates = await loop.run_in_executor(
                None, _verify_highlights, frames, candidates, str(settings.lora_adapter_path))
            version_tag += "+verified"      # A/B 구분 생성기 단독 vs 생성기 + 판별기

        for h in candidates:
            h["_model_version"] = version_tag
        logger.info(f"Phase 1  결과: {len(candidates)}개 후보")
        return candidates

    if settings.LORA_ENABLED and gen_path.exists():
        from app.services.phase2_inference import run_phase2_inference
        candidates = await run_phase2_inference(source_path, transcript_data, max_shorts, str(gen_path))
        logger.info(f"Phase 2 결과: {len(candidates)}개 후보")
        return candidates

    # 기존 경로: llama-server 서브프로세스
    frames = await _extract_default_frames(source_path, total_duration)
    proc = None
    external_server = _is_server_running()
    try:
        logger.info("VLM 분석 시작: llama-server 멀티모달 모드")
        if not external_server:
            proc = await loop.run_in_executor(None, start_llm_server, True)
        else:
            logger.info("외부 llama-server 감지 - 재사용")
        messages = _build_vlm_messages(frames, transcript_data, max_shorts)
        text = await loop.run_in_executor(None, _call_vlm_api, messages)
        return parse_highlights(text, total_duration, max_shorts, target_duration_sec=target_dur)
    finally:
        if proc is not None:
            await loop.run_in_executor(None, stop_llm_server, proc)

async def _extract_default_frames(source_path: str, total_duration: float) -> list[dict]:
    """기본 프레임 추출 (20프레임, 전체 영상 커버)"""

    dyn_interval = max(10.0, total_duration / 20) if total_duration > 200 else None
    frames = await extract_frames(Path(source_path), interval_sec=dyn_interval)
    if not frames:
        logger.warning("프레임 추출 결과 없음")
    return frames

# --------------------------------------------------------------
# 메시지 생성 - OpenAI 호환 멀티모달 포맷
# --------------------------------------------------------------

def _build_vlm_messages(frames: list[dict], transcript_data: dict, max_shorts: int) -> list[dict]:
    """VLM API용 OpenAI 호환 멀티모달 메시지 생성 (data URI base64)"""

    system_msg = {"role": "system", "content": "유튜브 쇼츠 편집 전문가. 영상 프레임 + 전사 분석하여 매력적인 쇼츠 구간을 선별. 반드시 JSON 형식으로만 응답."}
    content_parts = []
    for frame in frames[:5]:            # 최대 5장 (페이로드 크기 제한)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{frame['mime_type']};base64,{frame['base64']}",
            },
        })

    text_prompt = _build_vlm_text_prompt(frames, transcript_data, max_shorts)
    content_parts.append({"type": "text", "text": text_prompt})
    user_msg = {"role": "user", "content": content_parts}

    return [system_msg, user_msg]

def _build_vlm_text_prompt(frames: list[dict], transcript_data: dict, max_shorts: int) -> str:
    """VLM 텍스트 프롬프트 생성 (프레임 타임스탬프 + 전사 + 지시사항)"""

    parts = []

    # 영상 원본 제목 - 클릭 유발 제목 생성의 맥락으로 활용
    video_title = transcript_data.get("video_title", "")
    if video_title:
        parts.append(f"## 영상 원본 제목\n{video_title}\n")

    if frames:
        parts.append("## 영상 프레임 타임스탬프")
        for i, f in enumerate(frames):
            parts.append(f"프레임 {i + 1}: {f['timestamp_sec']}초 시점")
        parts.append("")

    segments = transcript_data.get("segments", [])
    if segments:
        parts.append("## 전사 텍스트 (타임스탬프 포함)")
        for seg in segments[:30]:
            text = seg.get("text", "").strip()
            if text:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                parts.append(f"[{start:.1f}~{end:.1f}초] {text}")
        parts.append("")
    else:
        parts.append("## 전사 텍스트: 음성 없음 (영상 프레임만으로 분석)")
        parts.append("")

    # 지시사항
    total_dur = transcript_data.get("duration_sec", 0)
    parts.append(f"영상 총 길이: {total_dur:.1f}초")
    parts.append(f"최대 쇼츠 수: {max_shorts}개")
    parts.append(f"쇼츠 길이 범위: {MIN_DURATION_SEC}~{MAX_DURATION_SEC}")
    parts.append("")
    target_dur = transcript_data.get("target_duration_sec")
    length_guide = (f"구간 길이: {target_dur}초에 인접하게 조정하세요. (사용자 지정)" if target_dur
                    else "구간 길이: 최소 30초 이상 권장. 30~60초(핵심) | 60~120초(심층 스토리). 10초 미만 금지")
    parts.append(f"영상 프레임 + 전사 종합 분석하여 매력적인 쇼츠 구간을 선별하세요.\n{length_guide}")
    parts.append("")
    parts.append(
        "title_suggestion 작성 규칙 (필수):\n"
        "- 반드시 한국어로 작성\n"
        "- 15자 이내의 짧고 강렬한 문구\n"
        "- 클릭을 유발하는 표현 사용\n"
        "- 좋은 예: '이게 말이 돼?', '진짜 1초 차이', '이 장면만 5번 봤다', '아무도 몰랐던 사실', '결국 이렇게 됨', "
        "'3번 죽을 뻔한 순간', '충격 반전', '역대급 순간', '못 믿겠지만...'\n"
        "- 금지: '하이라이트', '이거 다시 본구간', '영상_#N' - 이런 표현 절대 사용 금지\n"
        "- 영상 원본 제목의 맥락을 반영하되 그대로 복사하지 말 것\n"
        "- 쇼츠가 여러 개일 경우 각 제목을 서로 다르게 작성할 것\n"
        "- 숫자/이모지 활용 가능\n"
    )
    parts.append("")
    parts.append(
        'JSON 형식으로만 응답: {"highlights": [{"start_sec": N, "end_sec": N, "hook_score": N, '
        '"reason": "선정 이유", "title_suggestion": "클릭 유발 한글 제목", "tags": ["태그"], '
        '"recommended_aspect_ratio": "비율"}]}\n'
        "종횡비:9:16(인물/세로형 쇼츠) | 16:9(풍경/게임) | 1:1(인스타) | 4:5(인스타 세로) | 4:3(클래식)\n"
    )
    return "\n".join(parts)
    
# --------------------------------------------------------------
# VLM API 호출 - llama-server /v1/chat/completions
# --------------------------------------------------------------

def _call_vlm_api(messages: list[dict]) -> str:
    """llama-server /v1/chat/completions 호출"""

    url = f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}/v1/chat/completions"

    payload = json.dumps({
        "messages": messages,
        "temperature": 0.3,         # 하이라이트 추출은 정확성 우선
        "top_p": 0.95,              
        "max_tokens": 2000,
    }).encode("utf-8")

    logger.info(f"VLM API 호출 | {url} | 페이로드: {len(payload)}bytes")

    try:
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        resp = urlopen(req, timeout=120)
        body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"VLM API 호출 실패: {e}")
    
    # OpenAI 호환 응답에서 텍스트 추출
    try:
        result = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"VLM 응답 파싱 실패: {e} | body: {str(body)[:300]}")
    
    logger.info(f"VLM 응답 수신: {len(result)}자")
    return result

# --------------------------------------------------------------
# LoRA 추론 - Unsloth/Transformers 직접 (22일차, Qwen2.5-VL-7B)
# 모델 생명주기 헬퍼는 lora_utils.py 로 분리됨 (25일차)
# --------------------------------------------------------------

def _verify_highlights(frames: list[dict], candidates: list[dict], adapter_path: str) -> list[dict]:
    """판별기 LoRA로 각 후보의 적합성 검증 - scripts/verify_highlights.py 서브프로세스로 실행 (VRAM 분리)"""

    import subprocess, sys, json as _json, tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        _json.dump({"frames": frames, "candidates": candidates, "adapter_path": adapter_path}, f)
        tmp_path = f.name
    try:
        env = {**os.environ, "UNSLOTH_SUPPRESS_WARNINGS": "1", "PYTHONWARNINGS": "ignore"}
        result = subprocess.run(
            [sys.executable, "-m", "scripts.verify_highlights", tmp_path],
            capture_output=True, text=True, timeout=300, env=env
        )
        os.unlink(tmp_path)
        if result.returncode == 0:
            for line in reversed(result.stdout.strip().splitlines()):
                try:
                    verified = _json.loads(line)
                    logger.info(f"판별기: {len(verified)}/{len(candidates)}개 통과")
                    return verified
                except Exception:
                    continue
        logger.warning(f"판별기 서브프로세스 실패: {result.stderr[-200:]}")
        return candidates
    except Exception as e:
        logger.warning(f"판별기 실행 오류: {e}")
        return candidates

def _run_lora_inference(frames: list[dict], transcript_data: dict, max_shorts: int, adapter_path: str) -> str:
    """Unsloth LoRA 직접 추론 (동기, run_in_executor로 호출)"""

    model, tokenizer, processor = load_lora_model(adapter_path)
    images = frames_to_pil(frames)
    text_prompt = _build_vlm_text_prompt(frames, transcript_data, max_shorts)
    content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text_prompt}]
    result = lora_generate(model, tokenizer, processor, [{"role": "user", "content": content}], temp=0.4)
    unload_lora_model(model, tokenizer, processor)
    logger.info("LoRA 생성 완료")
    return result