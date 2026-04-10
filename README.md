# YT Shorts AI

로컬 GPU 가속 기반 유튜브 영상 분석 및 쇼츠 자동 편집 시스템

## 아키텍처 개요

```
app/
├── main.py                  # 진입점 (FastAPI 인스턴스, 미들웨어, 라우터 등록)
├── core/
│   ├── config.py            # 환경 설정 (Pydantic Settings)
│   ├── database.py          # 비동기 DB 연결 및 세션 관리
│   ├── security.py          # JWT 인증, 비밀번호 해싱
│   └── dependencies.py      # DI 체인 (컨트롤러→서비스→레포지토리→DB)
├── api/v1/
│   ├── router.py            # 라우터 집합
│   ├── projects.py          # 프로젝트 엔드포인트
│   ├── shorts.py            # 쇼츠 엔드포인트
│   └── system.py            # 시스템 상태 엔드포인트
├── models/
│   └── domain.py            # SQLModel 도메인 모델 (Project, Shorts)
├── schemas/
│   └── api.py               # Pydantic 요청/응답 스키마
├── repositories/
│   ├── base_repository.py   # 제네릭 CRUD 베이스
│   ├── project_repository.py
│   └── shorts_repository.py
├── services/
│   ├── video_service.py     # 다운로드 & 전처리
│   ├── analysis_service.py  # ASR & 하이라이트 추출
│   └── editing_service.py   # 리프레이밍 & 인코딩
├── workers/                 # 백그라운드 작업 (Celery 등)
└── utils/                   # 공통 유틸리티
```

## 의존성 주입 체인

```
API 엔드포인트 (컨트롤러)
  └─ Depends → Service (비즈니스 로직)
       └─ Depends → Repository (데이터 접근)
            └─ Depends → AsyncSession (DB 세션)
```

## 빠른 시작

```bash
# 1. 환경 구축 (WSL Ubuntu)
chmod +x setup_dev_env.sh
./setup_dev_env.sh

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키, 모델 경로 등 설정

# 4. 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. API 문서 확인
# http://localhost:8000/docs
```

## GPU 가속 PyTorch 설치 (별도)

```bash
# CUDA 12.4 기준
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2주 개발 로드맵

| 일차 | 주요 활동 |
|------|----------|
| 1-2 | 아키텍처 확립, 디렉토리 구조, DB 모델링 |
| 3-5 | 영상 다운로드, Whisper ASR, API 기초 |
| 6-7 | LLM 하이라이트 추출, 서비스 로직 |
| 8-10 | YOLOv8 리프레이밍, FFmpeg CUDA 필터 |
| 11-12 | 자막 합성, 테스트, 리팩토링 |
| 13-14 | 통합 테스트, 배포 준비, 문서화 |

## 테스트

```bash
pytest tests/ -v
```