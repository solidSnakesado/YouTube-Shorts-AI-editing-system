# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: 프레임/오디오/타겟을 Gemma messages 포맷 샘플로 조립 (Gemma 데이터 재구축 - 모듈 C-2a)
# 39일차 신규: 데이터셋 1행(= 1클립)을 Gemma conversation/messages JSON으로 생성
#   - 메인 빌더(gemma_dataset_builder)가 클립마다 호출
#   - content-block 키 스키마는 Colab 학습 노트북 로더에 맞춰 최종 조정 가능 (이 함수만 수정)
#   - 타겟 스키마를 hook_score만으로 축소 (타임스탬프/제목/상수 제거 -> 손실을 점수에 집중)

"""Gemma messages 샘플 빌더 - [1fps 프레임 + 30s 오디오 + 타겟] -> conversation JSON"""

import json

# 39일차: pos/neg 빌더 공유 상수 - instruction은 양쪽이 동일해야 학습이 일관됨, NEGATIVE_OUTPUT은 비피크 타겟
HIGHLIGHT_INSTRUCTION = (
    "영상 프레임과 오디오를 분석하여 이 클립이 쇼츠 하이라이트인지 판단하세요. "
    "하이라이트면 hook_score를 담은 highlights JSON으로, 아니면 빈 리스트로 반환하세요."
)
NEGATIVE_OUTPUT = '{"highlights": []}'      # 네거티브(비피크) 타겟 - 빈 하이라이트

def build_highlight_output(hook_score: float,) -> str:
    """타겟(assistant 출력) JSON 문자열 생성 - hook_score만 (39일차: 스키마 축소).

    hook_score는 히트맵 피크 avg_value(시청 재생률)를 그대로 사용한다.
    타임스탬프/제목/상수(reason/tags/aspect) 제거 이유:
        - 절대 타임스탬프: 클립 입력에서 절대 위치 추론 불가 -> 암기 유발
        - 상수 필드: 학습 신호 0이며 JSON 보일러플레이트가 cross-entropy를 지배해
          hook_score 신호를 희석 -> 제거 시 손실이 hook_score에 집중(collapse 저항)
    네거티브(비피크)는 빈 리스트 '{"highlights": []}'로 별도 생성한다.
    """

    return json.dumps({"highlights": [{"hook_score": round(hook_score, 4)}]}, ensure_ascii=False)

def build_gemma_sample(
    frame_paths: list[str],
    audio_path: str,
    instruction: str,
    output_json: str,
    metadata: dict,
) -> dict:
    """1클립 -> Gemma messages 포맷 샘플(dict) 조립
    
    user turn   : 1fps 프레임들(image) + 오디오(audio) + 지시문(text)
    assistant   : 하이라이트 JSON(text)
    미디어는 상대 경로로 참조하므로 Colab 업로드 후에도 동일하게 로드 가능

    주의: content-block 키('image'/'audio')는 Unsloth Gemma 오디오 노트북의 데이터
    로더 규격에 맞춰 학습 셋업 시 최종 확정 (이 함수만 수정하면 됨)

    Args:
        frame_paths: 1fps 프레임 상대 경로 리스트
        audio_path: 오디오 세그먼트 상대 경로
        instruction: 지시문 텍스트 (user)
        output_json: assistant 출력 (하이라이트 JSON 문자열)
        metadata: 추적용 메타데이터 (video_id, clip 구간 등)

    Returns:
        Gemma messages 포맷 샘플 dict
    """

    # user content: 프레임(image) N개 -> 오디오(audio) -> 지시문(text) 순
    user_content: list[dict] = [{"type": "image", "image": p} for p in frame_paths]
    user_content.append({"type": "audio", "audio": audio_path})
    user_content.append({"type": "text", "text": instruction})

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": output_json}]},
        ],
        "metadata": metadata,
    }