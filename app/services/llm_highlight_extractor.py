# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: LLM 프롬프트 생성, 호출, 응답 파싱 (analysis_service.py 300줄 규칙으로 분리)
# 의존: 없음 (gpu_manager가 반환한 LLM 핸들을 인자로 받아 사용)
# MVA 원칙: 인프라 책임(모델 로드/언로드)은 gpu_manager에 위임
# 흐름: transcript_json -> build_highlight_prompt -> call_llm -> parse_highlights
# 6~7일차 신규 / 11~12일차: create_time_based_highlight 추가
# 13일차: LLM 자동 길이 판단 - duration_sec 고정값 제거, 10~120초 범위 자동 결정
# 17일차: 청크 dict 허용 / chunk_start 파라미터 추가 / 청크 범위 밖 제외
# 21일차: recommended_aspect_ratio 필드 추가 / VALID_ASPECT_RATIOS 검증
# 24일차: _extract_json 4차 폴백 추가 (JSON 잘림 복구)

"""LLM 하이라이트 추출기 - 프롬프트 생성, LLM 호출, 응답 파싱"""

import json                     # LLM 응답 JSON 파싱
import re                       # 정규식으로 JSON 추출 (마크다운 코드블록 등)
from typing import Any

from loguru import logger

# 쇼츠 길이 제한 상수 - LLM이 자동 판단하되, 이 범위를 벗어나면 검증에서 보정
MIN_DURATION_SEC    = 10                                                # 최소 쇼츠 길이 (미만 시 제거)
MAX_DURATION_SEC    = 120                                               # 최대 쇼츠 길이 (초과 시 잘라내기)
VALID_ASPECT_RATIOS = {"9:16", "16:9", "1:1", "4:5", "4:3", "16:10"}    # 21일차

# --------------------------------------------------------------
# 프롬프트 생성
# --------------------------------------------------------------

def build_highlight_prompt(transcript_data: dict, max_shorts: int = 5) -> str:
    """전사 데이터를 LLM 프롬프트로 변환
    13일차 : LLM 자동 길이 판단 / 17일차: 청크 dict 도 허용 / 24일차: video_title 주입
    Args: transcript_data: Whisper 전사 결과 dict (segments, video_title 포함), max_shorts: 추출할 최대 쇼츠 수
    Returns: LLM에 전달할 프롬프트 문자열
    """

    # 세그먼트를 타임스탬프 포함 텍스트로 포맷팅 (절대 시각 유지)
    formatted_lines = []
    for seg in transcript_data.get("segments", []):
        start, end = seg.get("start", 0), seg.get("end", 0)
        text = seg.get("text", "").strip()
        if text:
            formatted_lines.append(f"[{start:.1f}-{end:.1f}s] {text}")

    transcript_text = "\n".join(formatted_lines)
    total_duration = transcript_data.get("duration_sec", 0)
    language = transcript_data.get("language", "unknown")

    # 17일차: 청크인 경우 절대 시각 범위 표시, 아니면 전체 영상 범위 사용
    start_offset = transcript_data.get("start_offset_sec", 0)
    end_offset = transcript_data.get("end_offset_sec", total_duration)
    video_title = transcript_data.get("video_title", "")
    title_context = f"\n    - 영상 원본 제목: {video_title}" if video_title else ""
    target_duration_sec = transcript_data.get("target_duration_sec")
    duration_rule = (
        f"각 클립 길이를 {target_duration_sec}초를 크게 벗어나지 않게 맞추세요. (사용자 지정)"
        if target_duration_sec else
        "각 클립은 최소 30초 이상으로 생성하세요. 30~60초(핵심) | 60~120초(심층 스토리). 10초 미만 금지"
    )

    return f"""당신은 유투브 쇼츠 편집 전문가입니다. 아래 영상의 전사 텍스트를 분석하여 가장 매력적인 쇼츠 클립 구간을 선별하세요.

    영상 정보:
        - 분석 범위: {start_offset:.0f}초 ~ {end_offset:.0f}초 (총 {total_duration:.0f}초)
        - 언어: {language}{title_context}
    선별 기준:
        - 최대 {max_shorts}개 클립 선별
        - {duration_rule}
        - 시청자의 관심을 끌수 있는 훅(Hook)이 있는 구간 우선
        - 맥락이 완결되는 구간 (문장 중간에 잘리지 않도록)
        - 감정적 반응을 유발하는 부분 (놀라움, 유머, 감동, 인사이트)
        - 시작 시점은 {start_offset:.0f}초 이상, 끝 시점은 {end_offset:.0f}초 이하
    title_suggestion 작성 규칙 (필수):
        - 반드시 한국어로 작성, 15자 이내의 짧고 강렬한 문구
        - 클릭을 유발하는 표현 사용 (예: '이게 실화?', '충격 반전', '역대급 순간')
        - 영상 원본 제목의 맥락을 반영하되 그대로 복사하지 말 것
        - 숫자/이모지 활용 가능 (예: '3번 죽을 뻔한 순간')
    반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요:
    {{
        "highlights": [
            {{
                "start_sec": 시작시간(초),
                "end_sec": 끝시간(초),
                "hook_score": 흥미도 점수(0.0~1.0),
                "reason": "선정 이유 (한국어, 1~2문장)",
                "title_suggestion": "클릭 유발 한글 제목 (15자 이내)",
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
    """LLM 호출 (OpenAI API / 로컬 Gemma 4 GGUF 분기, llm_handle["type"]으로 분기)"""

    llm_type = llm_handle.get("type")
    if llm_type == "openai":
        return _call_openai(llm_handle["client"], prompt)
    elif llm_type == "local":
        return _call_local(llm_handle["model"], prompt)
    else:
        raise ValueError(f"지원하지 않는 LLM 타입: {llm_type}")

def _call_openai(client: Any, prompt: str) -> str:
    """OpenAI API GPT-4o-mini 호출 (JSON 모드 강제, temperature=0.3)"""

    logger.info("OpenAI API 호출 시작")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "JSON 형식으로만 응답하세요."}, {"role": "user", "content": prompt}],
        temperature=0.3,                            # 낮은 temperature로 일관된 JSON 출력
        max_tokens=2000,
        response_format={"type": "json_object"},    # OpenAI JSON 모드 강제
    )

    result = response.choices[0].message.content
    logger.info(f"OpenAI 응답: {len(result)}자")
    return result

def _call_local(model: Any, prompt: str) -> str:
    """로컬 Gemma 4 E4B 호출 (llama-cpp-python, 권장 샘플링값 적용, temperature=0.3)"""

    logger.info("Gemma 4 E4B 로컬 호출 시작")
    response = model.create_chat_completion(
        messages=[{"role": "system", "content": "JSON 형식으로만 응답하세요."}, {"role": "user", "content": prompt}],
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

def parse_highlights(llm_response: str, total_duration: float, max_shorts: int, chunk_start: float = 0.0, target_duration_sec: int = 0) -> list[dict]:
    """
    LLM 응답에서 하이라이트 목록 추출 + 검증
    파싱 전략 (3단계 폴백): JSON 직접 -> 코드블록 -> 중괄호 패턴
    17일차: chunk_start 파라미터 추가 - 청크 범위 밖 하이라이느틑 제외 (오버랩 침범 방지)
    """

    raw = _extract_json(llm_response)
    if raw is None:
        logger.error(f"JSON 추출 실패: {llm_response[:200]}")
        return []
    
    if not isinstance(raw, dict):       # 33일차: json.loads는 JSON 배열도 허용 -> dict 아니면 .get 크래시 (방어 가드)
        logger.error(f"JSON 객체가 아님 (type: {type(raw).__name__}): {llm_response[:100]}")
        return []
    
    highlights_raw = raw.get("highlights", [])
    if not isinstance(highlights_raw, list):
        logger.error(f"highlights가 리스트가 아님: {type(highlights_raw)}")
        return []
    
    # 개별 항목 검증 (17일차: chunk_start 전달)
    validated = []
    for h in highlights_raw:
        item = _validate_highlight(h, total_duration, chunk_start, target_duration_sec)
        if item:
            validated.append(item)

    # hook_score 내림차순 정렬 후 max_shorts 개수 제한
    validated.sort(key=lambda x: x.get("hook_score", 0), reverse=True)
    result = validated[:max_shorts]

    logger.info(f"파싱 완료: {len(result)}개 (원본: {len(highlights_raw)}개)")
    return result

def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 객체를 추출 (3단계 폴백: 직접 -> 코드블록 -> 중괄호)"""

    # 1차: 전체 텍스트를 직접 JSON 파싱
    try: 
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 2차: ```json...``` 마크다운 코드 블록 내부 추출
    code_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
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
    # 4차: JSON 잘림 복구 - 완성된 하이라이트 객체만 추출 (24일차: max_new_tokens 초과 대응)
    item_matches = re.findall(r'\{[^{}]*"start_sec"[^{}]*"end_sec"[^{}]*\}', text, re.DOTALL)
    if item_matches:
        recovered = []
        for m in item_matches:
            try:
                recovered.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        if recovered:
            logger.warning(f"JSON 잘림 감지 - 완성된 {len(recovered)}개 항목으로 복구")
            return {"highlights": recovered}
    return None

def _validate_highlight(h: dict, total_duration: float, chunk_start: float = 0.0, target_duration_sec: int = 0) -> dict | None:
    """
    개별 하이라이트 검증 및 정규화
    검증: 시간 순서, 영상 범위 클리핑, 최소 5초, hook_score 0~1 클리핑
    Returns: 유효한 dict 또는 None
    13일차: MIN/MAX_DURATION_SEC 상수 적용, 초과 시 잘라내기 추가
    17일차: chunk_start 범위 검증 - 청크 범위 밖 하이라이트는 제외
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
    # 17일차: 청크 범위 빆 하이라이트 제외 (오버랩 영역에서 반대편 청크 침범 방지)
    if start < chunk_start or end > total_duration:
        logger.debug(
            f"청크 범위 밖 하이라이트 제외: [{start:.1f} - {end:.1f}] "
            f"청크 범위: [{chunk_start:.1f} - {total_duration:.1f}]"
        )
        return None
    # 최대 길이 초과 시 잘라내기 (13일차 추가)
    if end - start > MAX_DURATION_SEC:
        end = start + MAX_DURATION_SEC
    # 최소 길이 검증
    if end - start < MIN_DURATION_SEC:
        end = min(start + MIN_DURATION_SEC, total_duration)
        if end - start < MIN_DURATION_SEC:
            return None   
    if target_duration_sec > 0:
        end = min(start + target_duration_sec, total_duration)
        if end - start < MIN_DURATION_SEC:
            return None   
    
    # hook_score 클리핑 (0.0 ~ 1.0)
    try:
        score = max(0.0, min(1.0, float(h.get("hook_score", 0.5))))
    except (TypeError, ValueError):
        score = 0.5

    # aspect_ratio 검증 (21일차): 유효하지 않으면 기본값 "9:16"
    ar = str(h.get("recommended_aspect_ratio", "9:16")).strip()
    if ar not in VALID_ASPECT_RATIOS:
        ar = "9:16"

    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "hook_score": round(score, 4),
        "reason": str(h.get("reason", ""))[:500],                           # DB 컬럼 보호
        "title_suggestion": str(h.get("title_suggestion", ""))[:200],
        "tags": h.get("tags", [])[:10],                                     # 태그 최대 10개
        "aspect_ratio": ar,
    }

# --------------------------------------------------------------
# 음성 없는 영상용 시간 기반 풀백 (11~12일차 추가 / 13일차: duration_sec 제거)
# --------------------------------------------------------------
def create_time_based_highlights(total_duration: float, max_shorts: int) -> list[dict]:
    """음성이 없는 영상에서 시간 기간 균등 분할 (기본 60초 간격)"""

    if total_duration <= 0:
        return []
    default_dur = min(60, MAX_DURATION_SEC) # 기본 구간 길이
    interval = max(default_dur, total_duration / max_shorts)
    highlight = []
    for i in range(max_shorts):
        start = round(i * interval, 3)
        end = round(min(start + default_dur, total_duration), 3)
        if end - start < MIN_DURATION_SEC or start >= total_duration:
            break
        highlight.append({
            "start_sec": start, "end_sec": end,
            "hook_score": round(0.5 - i * 0.05, 4),
            "reason": f"시간 기반 자동 분할 #{i + 1} (음성 미감지)",
            "title_suggestion": f"{int(start//60):02d}분{int(start%60):02d}초_구간", "tags": [],
            "aspect_ratio": "9:16",
        })
    logger.info(f"시간 기반 하이라이트: {len(highlight)}개 (음성 없음)")
    return highlight