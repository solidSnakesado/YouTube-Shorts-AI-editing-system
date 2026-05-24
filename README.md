# YT Shorts AI

로컬 GPU 가속 기반 유튜브 영상 분석 및 쇼츠 자동 편집 시스템

## 아키텍처 개요

MVA (Minimum Viable Architecture) — 계층형 아키텍처 + 레포지토리 패턴 + 의존성 주입 체인

```
app/
├── main.py
├── core/
│   ├── config.py                    # 환경 설정 (Pydantic Settings, .env 바인딩)
│   ├── database.py                  # 비동기 DB 연결/세션
│   ├── security.py                  # JWT 인증, 비밀번호 해싱
│   ├── dependencies.py              # DI 체인 (API → 서비스 → 레포지토리 → DB)
│   ├── gpu_manager.py               # GPU VRAM 관리 (Whisper/LLM/YOLO 로드/언로드)
│   └── llm_server.py                # llama-server 서브프로세스 관리
├── api/v1/
│   ├── router.py
│   ├── projects.py                  # 프로젝트 엔드포인트
│   ├── shorts.py                    # 쇼츠 엔드포인트
│   ├── system.py                    # 시스템 상태 (GPU 모니터링)
│   └── heatmap.py                   # 히트맵 수집 엔드포인트
├── models/
│   └── domain.py                    # SQLModel 도메인 모델 (Project, Shorts)
├── schemas/
│   └── api.py                       # Pydantic 요청/응답 스키마
├── repositories/
│   ├── base_repository.py
│   ├── project_repository.py
│   └── shorts_repository.py
└── services/
    ├── video_service.py             # 다운로드 + 오디오 추출 (yt-dlp, FFmpeg)
    ├── analysis_service.py          # 전사(Whisper) + 하이라이트 추출
    ├── llm_highlight_extractor.py   # LLM 프롬프트/호출/파싱
    ├── vlm_client.py                # VLM 멀티모달 분석 (생성기→판별기 LoRA 파이프라인)
    ├── frame_extractor.py           # 영상 프레임 추출
    ├── dataset_builder.py           # LoRA 학습 데이터셋 생성
    ├── transcript_chunker.py        # 전사 청크 분할/재랭킹
    ├── editing_service.py           # 리프레이밍 + 자막 + 인코딩
    ├── reframe_engine.py            # 클립 추출 + YOLO 추적 + 적응형 크롭
    └── subtitle_generator.py        # ASS 자막 + FFmpeg 합성/인코딩

scripts/
    ├── build_finetune_dataset.py    # 학습 데이터 빌드 CLI
    ├── train_qlora.py               # QLoRA 파인튜닝 CLI
    ├── evaluate_lora.py             # LoRA 평가 CLI
    └── verify_highlights.py         # 판별기 LoRA 서브프로세스
```

## 7단계 파이프라인

```
유튜브 URL 입력
  → [1] 프로젝트 생성          POST /api/v1/projects/
  → [2] 영상 다운로드           POST /api/v1/projects/{id}/download
  → [3] 음성 전사 (Whisper)    POST /api/v1/projects/{id}/transcribe
  → [4] 하이라이트 추출 (LoRA) POST /api/v1/projects/{id}/analyze
  → [5] 리프레이밍 (YOLO)      POST /api/v1/shorts/{id}/edit
  → [6] 자막 합성 (ASS)        POST /api/v1/shorts/{id}/subtitle
  → [7] 최종 인코딩 (NVENC)    POST /api/v1/shorts/{id}/encode
  → outputs/{title}.mp4 (H.264, AAC)
```

## LoRA 파이프라인 (Step 4 상세)

```
LORA_ENABLED=true:
  생성기 LoRA (heatmap_generator) → 하이라이트 후보 JSON 생성
    → 판별기 LoRA (heatmap_adapter, 서브프로세스) → 후보 검증
      → 판별기 탈락 시 생성기 결과 반환 (방안 C)

LORA_ENABLED=false:
  llama-server (Qwen2.5-VL-7B GGUF) 기본 모드
```

## 빠른 시작

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치
uv sync

# 3. HuggingFace 로그인
uv run hf auth login

# 4. Qwen2.5-VL-7B GGUF 모델 다운로드 (~5GB)
uv run hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
    --include "*Q5_K_M*" --local-dir ./models/llm/

# 5. mmproj 다운로드 (~800MB)
uv run hf download unsloth/Qwen2.5-VL-7B-Instruct-GGUF \
    --include "*mmproj*" --local-dir ./models/llm/

# 6. Unsloth + AI 패키지 설치
uv pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
uv pip install unsloth_zoo bitsandbytes trl datasets pillow

# 7. llama.cpp 빌드 (GPU 아키텍처에 맞게 CUDA_ARCHITECTURES 조정)
# RTX 5070 Ti (Blackwell): 120 / RTX 4090 (Ada): 89 / RTX 3090 (Ampere): 86
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build llama.cpp/build --config Release -j4 --target llama-server
cp llama.cpp/build/bin/llama-server ./bin/llama-server

# 8. 환경변수 설정
cp .env.example .env
# .env 편집 후 LORA_ENABLED=false로 시작

# 9. 서버 실행
uv run uvicorn app.main:app --reload \
    --reload-exclude "unsloth_compiled_cache/*" \
    --host 0.0.0.0 --port 8000

# 10. 프론트엔드 (별도 터미널)
python3 -m http.server 3000
# http://localhost:3000/test.html
```

## LoRA 학습 (선택)

```bash
# 히트맵 수집
uv run python -m scripts.collect_heatmaps

# 판별기 학습 데이터 빌드
uv run python -m scripts.build_finetune_dataset \
    --heatmap data/heatmaps/heatmaps_2026-05-10.jsonl

# 판별기 LoRA 학습
uv run python -m scripts.train_qlora

# 생성기 학습 데이터 빌드
uv run python -m scripts.build_finetune_dataset \
    --heatmap data/heatmaps/heatmaps_2026-05-10.jsonl \
    --mode generator \
    --output data/finetune/dataset_generator.jsonl

# 생성기 LoRA 학습
uv run python -m scripts.train_qlora \
    --dataset data/finetune/dataset_generator_v2.jsonl \
    --adapter-type generator

# .env에서 LORA_ENABLED=true로 변경 후 서버 재시작
```

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | RTX 3080 (10GB) | RTX 5070 Ti (12GB) |
| RAM | 16GB | 32GB |
| OS | Ubuntu 22.04+ / WSL2 | WSL2 Ubuntu 24.04 |
| Python | 3.11.x | 3.11.x |
| CUDA | 12.x+ | 12.9 |
| FFmpeg | 6.x+ (NVENC, libass) | 최신 |

## VRAM 사용량 (순차 로딩)

```
[Step 3] Whisper medium        ~5GB  → 언로드
[Step 4] 생성기 LoRA (4bit)   ~5GB  → 언로드 → 판별기 LoRA 서브프로세스 ~5GB → 언로드
[Step 5] YOLOv8n               ~1GB  → 언로드
```

## 테스트

```bash
uv run pytest tests/ -v
```

## 개발 로드맵

| 일차 | 주요 활동 | 상태 |
|------|----------|:---:|
| 1-2 | 아키텍처 확립, DB 모델링 | ✅ |
| 3-5 | Whisper ASR, 다운로드 파이프라인 | ✅ |
| 6-7 | LLM 하이라이트 추출 | ✅ |
| 8-10 | YOLOv8 리프레이밍, FFmpeg 크롭 | ✅ |
| 11-12 | 자막 합성, 인코딩, 음성 없는 영상 대응 | ✅ |
| 13 | LLM 자동 길이 판단 | ✅ |
| 14-15 | VLM 멀티모달 분석 (llama-server) | ✅ |
| 17 | 청크 분할 LLM, 히트맵 수집 | ✅ |
| 21 | AI 종횡비 추천 | ✅ |
| 22 | 판별기 QLoRA 파인튜닝 (Qwen2.5-VL-7B) | ✅ |
| 23 | 생성기 QLoRA + 생성기→판별기 순차 파이프라인 | ✅ |
| 24 | 생성기 다수 후보 생성 개선 (D 방안) | 🔜 |
