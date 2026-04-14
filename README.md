# YT Shorts AI

로컬 GPU 가속 기반 유튜브 영상 분석 및 쇼츠 자동 편집 시스템

## 아키텍처 개요

MVA (Minimum Viable Architecture) — 계층형 아키텍처 + 레포지토리 패턴 + 의존성 주입 체인

```
app/
├── main.py                          # FastAPI 진입점 (미들웨어, 라우터, lifespan)
├── core/
│   ├── config.py                    # 환경 설정 (Pydantic Settings, .env 바인딩)
│   ├── database.py                  # 비동기 DB 연결/세션 (HTTPException 커밋 전략)
│   ├── security.py                  # JWT 인증, 비밀번호 해싱
│   ├── dependencies.py              # DI 체인 (API → 서비스 → 레포지토리 → DB)
│   └── gpu_manager.py               # GPU VRAM 관리 (Whisper/LLM/YOLO 로드/언로드)
├── api/v1/
│   ├── router.py                    # 라우터 집합
│   ├── projects.py                  # 프로젝트 엔드포인트 (create/download/transcribe/analyze)
│   ├── shorts.py                    # 쇼츠 엔드포인트 (edit/subtitle/encode)
│   └── system.py                    # 시스템 상태 (GPU 모니터링)
├── models/
│   └── domain.py                    # SQLModel 도메인 모델 (Project, Shorts)
├── schemas/
│   └── api.py                       # Pydantic 요청/응답 스키마
├── repositories/
│   ├── base_repository.py           # 제네릭 CRUD 베이스
│   ├── project_repository.py        # 프로젝트 저장소
│   └── shorts_repository.py         # 쇼츠 저장소
└── services/
    ├── video_service.py             # 다운로드 + 오디오 추출 (yt-dlp, FFmpeg)
    ├── analysis_service.py          # 전사(Whisper) + 하이라이트 추출(LLM)
    ├── llm_highlight_extractor.py   # LLM 프롬프트/호출/파싱 + 시간 기반 폴백
    ├── editing_service.py           # 리프레이밍 + 자막 합성 + 인코딩
    ├── reframe_engine.py            # 클립 추출 + YOLO 추적 + 적응형 크롭
    └── subtitle_generator.py        # ASS 카라오케 자막 + FFmpeg 합성/인코딩
```

## 7단계 파이프라인

```
유튜브 URL 입력
  → [1] 프로젝트 생성          POST /api/v1/projects/
  → [2] 영상 다운로드           POST /api/v1/projects/{id}/download
  → [3] 음성 전사 (Whisper)    POST /api/v1/projects/{id}/transcribe
  → [4] 하이라이트 추출 (LLM)  POST /api/v1/projects/{id}/analyze
  → [5] 리프레이밍 (YOLO)      POST /api/v1/shorts/{id}/edit
  → [6] 자막 합성 (ASS)        POST /api/v1/shorts/{id}/subtitle
  → [7] 최종 인코딩 (NVENC)    POST /api/v1/shorts/{id}/encode
  → outputs/{shorts_id}.mp4 (1080x1920, 9:16, H.264, AAC)
```

## 빠른 시작

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치
cd ~/projects/yt-shorts-ai
uv sync

# 3. llama-cpp-python CUDA 빌드 (RTX 5070 Ti / Blackwell)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 uv add llama-cpp-python --no-cache

# 4. 환경변수 설정
cp .env.example .env
# .env에서 LLM_MODEL_NAME, OPENAI_API_KEY 등 설정

# 5. Gemma 4 E4B 모델 다운로드 (~8-9GB)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/gemma-4-E4B-it-GGUF', allow_patterns=['*Q8_0*'], local_dir='./models/llm/')
"

# 6. 서버 실행
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 7. 프론트엔드 테스트 페이지 (별도 터미널)
python3 -m http.server 3000
# http://localhost:3000/test.html

# API 문서: http://localhost:8000/docs
```

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | GTX 1080 (8GB) | RTX 5070 Ti (12GB) |
| RAM | 16GB | 32GB |
| OS | Ubuntu 22.04+ / WSL2 | WSL2 Ubuntu 24.04 |
| Python | 3.11.x | 3.11.x |
| CUDA | 12.x+ | 13.x |
| FFmpeg | 6.x+ (NVENC, libass) | 최신 |

## VRAM 사용량 (순차 로딩)

```
[Step 3] Whisper medium ~5GB → 언로드
[Step 4] Gemma 4 E4B Q8_0 ~8-9GB → 언로드
[Step 5] YOLOv8n ~1-2GB → 언로드
```

## 테스트

```bash
# 전체 테스트 (18개)
uv run pytest tests/ -v

# 개별 테스트
uv run pytest tests/integration/test_encoding.py -v     # 자막/인코딩
uv run pytest tests/integration/test_reframe.py -v      # 리프레이밍
uv run pytest tests/integration/test_highlights.py -v   # 하이라이트
uv run pytest tests/integration/test_pipeline.py -v     # 다운로드/전사
```

## 개발 로드맵

| 일차 | 주요 활동 | 상태 |
|------|----------|:---:|
| 1-2 | 아키텍처 확립, DB 모델링, 디렉토리 구조 | ✅ |
| 3-5 | Whisper ASR, 다운로드 파이프라인 | ✅ |
| 6-7 | LLM 하이라이트 추출 (Gemma 4 E4B) | ✅ |
| 8-10 | YOLOv8 리프레이밍, FFmpeg 크롭 | ✅ |
| 11-12 | 자막 합성, 인코딩, 음성 없는 영상 대응 | ✅ |
| 13-14 | E2E 재테스트, 쇼츠 길이 LLM 판단, 제목 삽입 | 🔜 |
| 15-16 | VLM 영상+텍스트 통합 분석, 레이아웃 선택 | 🔜 |
