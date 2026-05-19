# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: VLM(Vision-Language Model) 멀티모달 분석의 전체 오케스트레이션
#       llama-server 시작 -> 프레임 추출 -> VLM API 호출 -> 응답 파싱 -> 서버 종료
#       analysis_service.py의 300줄 규칙 준수를 위해 분리된 헬퍼 모듈
# 의존: 
#   - app.core.config (서버 포트, 프레임 설정)
#   - app.core.llm_server (서버 시작/종료)
#   - app.services.frame_extractor (영상 프레임 추출)
#   - app.services.llm_highlight_extractor (parse_highlight 재사용)
# MVA 원칙: VLM 호출 로직은 서비스 헬퍼, GPU/서버 관리는 인프라에 위임
#
# 14~15일차 신규: 
#   - is_vlm_available(): VLM 모드 사용 가능 여부 확인
#   - run_vlm_analysis(): 전체 VLM 분석 파이프라인 오케스트레이션 (async)
#   - _build_vlm_messages(): OpenAI 호환 멀티모달 메시지 생성
#   - _build_vlm_text_prompt(): 프레임 타임스탬프 + 전사 텍스트 프롬프트
#   - _call_vlm_api(): llama-server HTTP API 호출
#
# 호출: analysis_service -> is_vlm_available() -> run_vlm_analysis()
#   내부: start_llm_server -> extract_frames -> _build_vlm_messages -> _call_vlm_api -> parse_highlight -> stop_llm_server

"""
VLM 클라이언트 - 영상 프레임 + 텍스트 통합 분석 오케스트레이터

텍스트만 분석하던 기존 LLM 파이프라인을 대체하여 영상 프레임과 전사 텍스트를 동시에 VLM에 전달하여
시각적 + 언어적 하이라이트 추출
"""

import asyncio
import json
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

from app.core.config import settings
from app.core.llm_server import start_llm_server, stop_llm_server
from app.services.frame_extractor import extract_frames

# parse_highlights 재사용: VLM 응답도 동일한 JSON 구조이므로 기존 파서 활용
from app.services.llm_highlight_extractor import parse_highlights, MIN_DURATION_SEC, MAX_DURATION_SEC

def is_vlm_available() -> bool:
    """
    VLM 멀티모달 분석 사용 가능 여부 확인

    조건: llama-server 바이너리 + GGUF 모델 + mmproj 프로젝터가 모두 존재
    하나라도 없으면 False -> 텍스트 전용 LLM 폴백

    Returns:
        True: VLM 사용 가능, False: 텍스트 전용 폴백
    """

    server_ok = Path(settings.LLAMA_SERVER_PATH).is_file()
    mmproj_ok = settings.mmproj_model_file.is_file()

    if not server_ok:
        logger.debug("VLM 불가: llama-server 바이너리 미존재")
    if not mmproj_ok:
        logger.debug("VLM 불가: mmproj 파일 미존재")

    return server_ok and mmproj_ok

async def run_vlm_analysis(source_path: str, transcript_data: dict, max_shorts: int = 5) -> list[dict]:
    """
    VLM 멀티모달 분석 전체 파이프라인 실행
    영상 프레임 + 전사 텍스트를 VLM에 전달하여 시각 + 언어 기반 하이라이트 추출
    음성이 없는 영상에서도 시각 정보만으로 하이라이트 선정 가능

    Args:
        source_path: 소스 영상 파일 경로 (temp/{pid}/source.mp4)
        transcript_data: Whisper 전사 결과 dict (segments 포함)
        max_shorts: 추출할 최대 쇼츠 수
    
    Returns:
        parse_highlights()가 반환하는 검증된 하이라이트 목록
    """

    loop = asyncio.get_event_loop()
    proc = None

    try:
        # 1. llama-server 시작 (멀티모달 모드) - 동기 -> 스레드 풀
        logger.info("VLM 분석 시작: llama-server 멀티모달 모드")
        proc = await loop.run_in_executor(None, start_llm_server, True)

        # 2. 영상 프레임 추출 - 비동기 (FFmpeg 서브프로세스)
        frames = await extract_frames(Path(source_path))
        if not frames:
            logger.warning("프레임 추출 결과 없음 - 텍스트 전용 프롬프트로 폴백")

        # 3. VLM 메시지 생성 (이미지 + 텍스트 통합)
        messages = _build_vlm_messages(frames, transcript_data, max_shorts)

        # 4. VLM API 호출 - 동기(urllib) -> 스레드 풀
        response_text = await loop.run_in_executor(None, _call_vlm_api, messages)

        # 응답 파싱 (기존 parse_highlights 재사용)
        total_duration = transcript_data.get("duration_sec", 0)
        highlights = parse_highlights(response_text, total_duration, max_shorts)

        logger.info(f"VLM 분석 완료: {len(highlights)}개 하이라이트 추출")
        return highlights
    finally:
        # 6. llama-server 종료 (VRAM 자동 해제)
        if proc is not None:
            await loop.run_in_executor(None, stop_llm_server, proc)

# --------------------------------------------------------------
# 메시지 생성 - OpenAI 호환 멀티모달 포맷
# --------------------------------------------------------------

def _build_vlm_messages(frames: list[dict], transcript_data: dict, max_shorts: int) -> list[dict]:
    """
    VLM API용 OpenAI 호환 멀티모달 메시지 생성
    이미지는 data URI base64로, 텍스트는 content 배열로 전달
    구조: [system: JSON 지시, user: [image1, image2, ..., text_prompt]]
    """

    system_msg = {
        "role": "system",
        "content": (
            "당신은 유튜브 쇼츠 편집 전문가입니다. "
            "영상 프레임과 전사 텍스트를 분석하여 "
            "가장 매력적인 쇼츠 구간을 선별합니다. "
            "반드시 JSON 형식으로만 응답하세요. "
        ),
    }

    # user 메시지: 이미지 + 텍스트를 content 배열로 구성
    content_parts = []

    # 이미지 파트 - 각 프레임을 data URI로 전달
    for frame in frames:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{frame['mime_type']};base64,{frame['base64']}",
            },
        })

    # 텍스트 파트 - 프레임 타임스탬프 + 전사 텍스트 + 지시사항
    text_prompt = _build_vlm_text_prompt(frames, transcript_data, max_shorts)
    content_parts.append({"type": "text", "text": text_prompt})

    user_msg = {"role": "user", "content": content_parts}

    return [system_msg, user_msg]

def _build_vlm_text_prompt(frames: list[dict], transcript_data: dict, max_shorts: int) -> str:
    """
    VLM 텍스트 프롬프트 생성 - 프레임 설명 + 전사 텍스트 + 지시사항

    기존 build_highlight_prompt()와 유사하나,
    프레임 타임스탬프 매핑 정보가 추가되어 VLM이 시각과 텍스트를 연결 가능
    """

    parts = []

    # 프레임 타임스탬프 매핑 (VLM이 이미지와 시간을 연결할 수 있도록)
    if frames:
        parts.append("## 영상 프레임 타임스탬프")
        for i, f in enumerate(frames):
            parts.append(f"프레임 {i + 1}: {f['timestamp_sec']}초 시점")
        parts.append("")

    # 전사 텍스트 (타임스탬프 포함)
    segments = transcript_data.get("segments", [])
    if segments:
        parts.append("## 전사 텍스트 (타임스탬프 포함)")
        for seg in segments:
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
    parts.append(f"## 분석 지시사항")
    parts.append(f"영상 총 길이: {total_dur:.1f}초")
    parts.append(f"최대 쇼츠 수: {max_shorts}개")
    parts.append(f"쇼츠 길이 범위: {MIN_DURATION_SEC}~{MAX_DURATION_SEC}")
    parts.append("")
    parts.append(
        "위 영상 프레임과 전사 텍스트를 종합 분석하여 "
        "가장 매력적인 쇼츠 구간을 선별하세요.\n"
        "시각적으로 흥미로운 장면(표정 변화, 액션, 시각효과)과 "
        "언어적으로 임팩트 있는 구간(핵심 발언, 반전, 유머)을 "
        "모두 고려하세요.\n"
        "각 구간의 최적 길이를 콘텐츠에 맞게 자동 판단하세요:\n"
        "   - 10~15초: 임팩트 한 장면 (리액션, 반전)\n"
        "   - 15~30초: 핵심 요약 (인사이트, 팁)\n"
        "   - 30~60초: 스토리텔링 (맥락 필요)\n"
        "   - 60~120초: 심층 콘텐츠 (강의, 설명)\n"
    )
    parts.append("")
    parts.append(
        '반드시 아래 JSON 형식으로만 응답:\n'
        '{"highlights": [\n'
        '   {"start_sec": 0.0, "end_sec": 30.0, "hook_score": 0.95,\n'
        '    "reason": "선정 이유", "title_suggestion": "제목",\n'
        '    "tags": ["태그1", "태그2"],\n'
        '    "recommended_aspect_ratio": "9:16"}\n'
        "]}\n\n"
        "종횡비 선택 기준:\n"
        "- 9:16: 인물 중심, 세로형 쇼츠/릴스/틱톡\n"
        "- 16:9: 풍경, 게임플레이, 시네마틱, 다수 인물\n"
        "- 1:1: 인스타그램 피드, 대칭 구도\n"
        "- 4:5: 인스타그램 세로, 인물+배경 균형\n"
        "- 4:3: 클래식, 레트로, 프레젠테이션\n"
        "- 16:10: 와이드 게임플레이, 울트라 와이드\n"
    )

    return "\n".join(parts)
    
# --------------------------------------------------------------
# VLM API 호출 - llama-server /v1/chat/completions
# --------------------------------------------------------------

def _call_vlm_api(messages: list[dict]) -> str:
    """
    llama-server의 OpenAI 호환 API(/v1/chat/completions)로 VLM 추론 요청
    동기 함수 - run_in_executor로 호출, 타임아웃 120초 (이미지 다수 시 추론 시간 증가)
    """

    url = f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}/v1/chat/completions"

    payload = json.dumps({
        "messages": messages,
        "temperature": 0.3,         # 하이라이트 추출은 정확성 우선
        "top_p": 0.95,              # Gemma 4 권장값
        "top_k": 64,                # Gemma 4 권장값
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

        # 타임 아웃 120초: 이미지 다수 포함 시 추론 시간 증가
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