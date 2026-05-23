# 계층: 스크립트 (CLI 진입점)
# 역할: 판별기 LoRA 하이라이트 후보 검증 - vlm_client.py에서 subprocess로 호출
# 의존: vlm_client.py (_load_lora_model, _frames_to_pil, _lora_generate, _unload_lora_model)
# 23일차 신규: VRAM 완전 해제를 위해 별도 프로세스로 분리
#
# 사용법 (vlm_client.py 내부에서 자동 호출)
#   python -m scripts.verify_highlights /tmp/verify_input.json

"""판별기 LoRA 서브 프로세스 - 하이라이트 후보 검증 후 JSON 결과 stdout 출력"""

import json
import sys
from pathlib import Path

def main() -> None:
    """임시 파일에서 입력 읽기 -> 판별기 LoRA 검증 -> stdout JSON 출력"""

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.verify_highlights <input.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"입력 파일 없음: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    candidates = data["candidates"]
    adapter_path = data["adapter_path"]

    # vlm_client.py 헬퍼 함수 import
    from app.services.vlm_client import (
        _load_lora_model, _frames_to_pil, _lora_generate, _unload_lora_model,
    )

    model, tokenizer, processor = _load_lora_model(adapter_path)
    images = _frames_to_pil(frames)

    verified = []
    for hl in candidates:
        prompt = (
            f"구간 {hl.get('start_sec', 0):.1f}~{hl.get('end_sec', 0):.1f}초가 "
            f"하이라이트인지 판단하세요."
        )
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        result = _lora_generate(
            model, tokenizer, processor,
            [{"role": "user", "content": content}],
            max_tokens=50, temp=0.1
        )
        if "하이라이트" in result or "true" in result.lower():
            verified.append(hl)

    _unload_lora_model(model, tokenizer, processor)

    # 결과를 stdout JSON으로 출력 (vlm_client.py에서 파싱)
    print(json.dumps(verified, ensure_ascii=False))

if __name__ == "__main__":
    main()