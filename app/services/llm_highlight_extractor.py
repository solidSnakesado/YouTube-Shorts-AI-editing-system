# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: LLM 프롬프트 생성, 호출, 응답 파싱 (analysis_service.py 300줄 규칙으로 분리)
# 의존: 없음 (gpu_manager가 반환한 LLM 핸들을 인자로 받아 사용)
# MVA 원칙: 인프라 책임(모델 로드/언로드)은 gpu_manager에 위임
# 흐름: transcript_json -> build_highlight_prompt -> call_llm -> parse_highlights
# 6~7일차 신규 / 11~12일차: create_time_based_highlight 추가

"""LLM 하이라이트 추출기 - 프롬프트 생성, LLM 호출, 응답 파싱"""

import json                     # LLM 응답 JSON 파싱
import re                       # 정규식으로 JSON 추출 (마크다운 코드블록 등)
from typing import Any

from loguru import logger

# --------------------------------------------------------------
# 프롬프트 생성
# --------------------------------------------------------------

def build_highlight_prompt(transcript_data: dict, max_shorts: int = 5, duration_sec: int = 60) -> str:
    """
    전사 데이터를 LLM 프롬프트로 변환

    각 세그먼트를 [시작-끝초] 텍스트 형식으로 포맷팅하고
    LLM에게 흥미 구간 선별을 요청하는 프롬프트를 생성

    프롬프트 설계 의도:
        - 역할 부여: "유투브 쇼츠 편집 전문가"
        - 영상 정보 제공: 총 길이, 언어
        - 선별 기준 명시: 훅, 맥락 완결, 감정 반응
        - 출력 형식 강제: JSON only (다른 텍스트 금지)
        - 시간 범위 제한: 0초 - total_duration초
    Args:
        transcript_data: Whisper 전사 결과 dict (segments 포함)
        max_shorts: 추출할 최대 쇼츠 수
        duration_sec: 각 쇼츠의 목표 길이 (초)

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 세그먼트를 타임스탬프 포함 텍스트로 포맷팅
    # ex> "[0.0~3.5s] 안녕하세요 오늘은 놀라운 이야기를 해보겠습니다"
    formatted_lines = []
    for seg in transcript_data.get("segments", []):
        start, end = seg.get("start", 0), seg.get("end", 0)
        text = seg.get("text", "").strip()
        if text:
            formatted_lines.append(f"[{start:.1f}-{end:.1f}s] {text}")

    transcript_text = "\n".join(formatted_lines)
    total_duration = transcript_data.get("duration_sec", 0)
    language = transcript_data.get("language", "unknown")

    return f"""당신은 유투브 쇼츠 편집 전문가입니다.
    아래 영상의 전사 텍스트를 분석하여 가장 매력적인 쇼츠 클립 구간을 선별하세요.

    영상 정보:
        - 총 길이: {total_duration:.0f}초
        - 언어: {language}
    선별 기준:
        - 각 클립은 약 {duration_sec}초 내외로 구성
        - 최대 {max_shorts}개 클립 선별
        - 시청자의 관심을 끌수 있는 훅(Hook)이 있는 구간 우선
        - 맥락이 완결되는 구간 (문장 중간에 잘리지 않도록)
        - 감정적 반응을 유발하는 부분 (놀라움, 유머, 감동, 인사이트)
        - 시작 시점은 0초 이상, 끝 시점은 {total_duration:.0f}초 이하
    반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요:
    {{
        "highlights": [
            {{
                "start_sec": 시작시간(초),
                "end_sec": 끝시간(초),
                "hook_score": 흥미도 점수(0.0~1.0),
                "reason": "선정 이유 (한국어, 1~2문장)",
                "title_suggestion": "쇼츠 제목 제안 (한국어)",
                "tags": ["태그1", "태그2", "태그3"]
            }}
        ]
    }}

    전사 텍스트
    {transcript_text}"""

# --------------------------------------------------------------
# LLM 호출
# --------------------------------------------------------------

def call_llm(llm_handle: dict, prompt: str) -> str:
    """
    LLM 호출 (OpenAI API / 로컬 Gemma 4 GGUF 분기)

    llm_handle의 "type"키로 분기:
        "openai"    -> OpenAI GPT-4o-mini API 호출
        "local"     -> llama-cpp-python으로 로컬 Gemma 4 E4B 호출
    Args:
        llm_handle: gpu_manager.load_llm()이 반환한 핸들 dict
        prompt: LLM에 전달할 프롬프트
    Returns:
        LLM 응답 텍스트 (JSON 문자열 이어야 함)
    """

    llm_type = llm_handle.get("type")

    if llm_type == "openai":
        return _call_openai(llm_handle["client"], prompt)
    elif llm_type == "local":
        return _call_local(llm_handle["model"], prompt)
    else:
        raise ValueError(f"지원하지 않는 LLM 타입: {llm_type}")
    

def _call_openai(client: Any, prompt: str) -> str:
    """
    OpenAI API를 통한 LLM 호출
    GPT-4o-mini 사용 (비용 효율적, JSON 출력 우수)
    response_format=json_object로 JSON 출력 강제
    """

    logger.info("OpenAI API 호출 시작")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "JSON 형식으로만 응답하세요."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,                            # 낮은 temperature로 일관된 JSON 출력
        max_tokens=2000,
        response_format={"type": "json_object"},    # OpenAI JSON 모드 강제
    )

    result = response.choices[0].message.content
    logger.info(f"OpenAI 응답: {len(result)}자")
    return result

def _call_local(model: Any, prompt: str) -> str:
    """
    로컬 Gemma 4 E4B 호출 (llama-cpp-python)

    Gemma 4 공식 권장 샘플링 파라미터:
        temperature=1.0, top_p=0.95, top_k=64, repeat_penalty=1.0
    하이라이트 추출은 정확한 JSON이 필요하므로:
        temperature를 0.3으로 낮춤 (창의성 < 정확성)
        나머지는 Gemma 4 권장값 유지
    """

    logger.info("Gemma 4 E4B 로컬 호출 시작")

    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "JSON 형식으로만 응답하세요."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,                            # 낮은 temperature로 일관된 JSON 출력
        top_p=0.95,                                 # Gemma 4 권장값
        top_k=64,                                   # Gemma 4 권장값
        repeat_penalty=1.0,                         # Gemma 4 권장값
        max_tokens=2000,
    )

    result = response["choices"][0]["message"]["content"]
    logger.info(f"Gemma 4 응답: {len(result)}자")
    return result

# --------------------------------------------------------------
# 응답 파싱
# --------------------------------------------------------------

def parse_highlights(llm_response: str, total_duration: float, max_shorts: int) -> list[dict]:
    """
    LLM 응답에서 하이라이트 목록 추출 + 검증
    파싱 전략 (3단계 폴백): JSON 직접 -> 코드블록 -> 중괄호 패턴
    검증: 시간 범위, 최소 5초, hook_score 클리핑, max_shorts 제한
    """

    raw = _extract_json(llm_response)
    if raw is None:
        logger.error(f"JSON 추출 실패: {llm_response[:200]}")
        return []
    
    highlights_raw = raw.get("highlights", [])
    if not isinstance(highlights_raw, list):
        logger.error(f"highlights가 리스트가 아님: {type(highlights_raw)}")
        return []
    
    # 개별 항목 검증
    validated = []
    for h in highlights_raw:
        item = _validate_highlight(h, total_duration)
        if item:
            validated.append(item)

    # hook_score 내림차순 정렬 후 max_shorts 개수 제한
    validated.sort(key=lambda x: x.get("hook_score", 0), reverse=True)
    result = validated[:max_shorts]

    logger.info(f"파싱 완료: {len(result)}개 (원본: {len(highlights_raw)}개)")
    return result

def _extract_json(text: str) -> dict | None:
    """
    텍스트에서 JSON 객체를 추출 (3단계 폴백)
    LLM이 순수 JSON 대신 마크다운 코드블록이나
    설명 텍스트를 함께 출력하는 경우를 대응
    """

    # 1차: 전체 텍스트를 직접 JSON 파싱
    try: 
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2차: ```json...``` 마크다운 코드 블록 내부 추출
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3차: 첫 번째 { ~ 마지막 } 범위 추출
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None

def _validate_highlight(h: dict, total_duration: float) -> dict | None:
    """
    개별 하이라이트 검증 및 정규화
    검증: 시간 순서, 영상 범위 클리핑, 최소 5초, hook_score 0~1 클리핑
    Returns: 유효한 dict 또는 None
    """

    # 시간 값 파싱
    try:
        start = float(h.get("start_sec", -1))
        end = float(h.get("end_sec", -1))
    except (TypeError, ValueError):
        return None
    
    # 시간 순서 검증
    if start < 0 or end <= start:
        return None
    
    # 영상 범위 내로 클리핑
    start = max(0.0, start)
    end = min(total_duration, end)

    # 최소 길이 검증 (5초 미만이면 쇼츠로 부적합)
    if end - start < 5.0:
        return None
    
    # hook_score 클리핑 (0.0 ~ 1.0)
    try:
        score = max(0.0, min(1.0, float(h.get("hook_score", 0.5))))
    except (TypeError, ValueError):
        score = 0.5

    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "hook_score": round(score, 4),
        "reason": str(h.get("reason", ""))[:500],                           # DB 컬럼 보호
        "title_suggestion": str(h.get("title_suggestion", ""))[:200],
        "tags": h.get("tags", [])[:10],                                     # 태그 최대 10개
    }

# --------------------------------------------------------------
# 음성 없는 영상용 시간 기반 풀백 (11~12일차 추가)
# --------------------------------------------------------------
def create_time_based_highlights(total_duration: float, max_shorts: int, duration_sec: int) -> list[dict]:
    """
    음성이 없는 영상에서 시간 기간 균등 분할 하이라이트 생성
    LLM 호출 없이 영상을 duration_sec 길이로 균등 분할
    """

    if total_duration <= 0:
        return []
    
    highlight = []
    interval = max(duration_sec, total_duration / max_shorts)
    for i in range(max_shorts):
        start = round(i * interval, 3)
        end = round(min(start + duration_sec, total_duration), 3)
        if end - start < 5.0 or start >= total_duration:
            break
        highlight.append({
            "start_sec": start, "end_sec": end,
            "hook_score": round(0.5 - i * 0.05, 4),
            "reason": f"시간 기반 자동 분할 #{i + 1} (음성 미감지)",
            "title_suggestion": f"하이라이트 #{i + 1}", "tags": [],
        })
    logger.info(f"시간 기반 하이라이트: {len(highlight)}개 (음성 없음)")
    return highlight