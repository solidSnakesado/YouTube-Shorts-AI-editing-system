# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: Whisper 단어 타임스탬프 기반 ASS 동적 자막 생성, FFmpeg 자막/인코딩 실행
#       editing_service.py의 300줄 규칙으로 분리된 모듈
# 의존: 없음 (전사 데이터와 파일 경로를 인자로 받아 사용)
# MVA 원칙: 인프라 책임(FFmpeg 실행)은 비동기 서브프로세스로 처리
# 흐름: transcript -> extract_words_for_range -> build_add_header
#       -> build_ass_events -> write_ass_file -> run_ffmpeg_subtitle -> run_ffmpeg_encode
# 11~12일차 신규파일
# 17일차 변경사항:
#   - FONT_SIZE: 18 -> 52 (1080x1920 가독성 확보)
#   - FONT_SIZE_HL: 22 -> 62 (강조 단어 상대 강조)
#   - MARGIN_V: 120 -> 180 (하단 여백 확대)
#   - Outline: 3 -> 4, Shadow: 1 -> 2 (외곽선/그림자 강화)
#   - verify_font() 함수 신규 추가 (fc-match 기반 폰트 존재 검증)

"""
ASS 자막 생성기 - Whisper 타임스탬프 기반 동적 자막, 단어 강조, FFmpeg 합성/인코딩

ASS(Advenced Substation Alpha) 포맷 사용 이유:
    - SRT보다 풍부한 스타일링 (위치, 색상, 크기, 애니메이션)
    - 현재 발화 단어를 실시간 강조하는 카라오케 효과 지원
    - FFmpeg subtitles/ass 필터로 직접 합성 가능
"""

import asyncio
import subprocess
from pathlib import Path

from loguru import logger
from app.core.config import settings

# --- ASS 스타일 상수 ---
FONT_NAME = "Noto Sans CJK KR"      # 한국어 지원 무료 폰트, 시스템 설치 필수
FONT_SIZE = 52                      # 기본 폰트 크기 (1080x1920), 17일차: 18 -> 52
FONT_SIZE_HL = 62                   # 강조 단어 폰트 크기, 17일차: 22 -> 62
CLR_PRIMARY = "&H00FFFFFF"          # 기본: 흰색 (ASS BGR 형식)
CLR_HIGHLIGHT = "&H0000FFFF"        # 강조: 노란색 
CLR_OUTLINE = "&H00000000"          # 외곽선: 검은색 
CLR_SHADOW = "&H80000000"           # 그림자: 반투명 검은색 
MARGIN_V = 180                      # 하단 여백 (픽셀), 17일차: 120 -> 180 (하단 여백)
OUTLINE_WIDTH = 4                   # 17일차: 3 -> 4 (외곽선 두께)
SHADOW_DEPTH = 2                    # 17일차: 1 -> 2 (그림자 깊이)
WORDS_PER_GROUP = 4                 # 한 번에 표시할 단어 수

# --------------------------------------------------------------
# 0. 폰트 존재 검증 (17일차 신규)
# --------------------------------------------------------------
def verify_font(font_name: str = FONT_NAME) -> bool:
    """
    시스템에 지정 폰트가 설치되어 있는지 fc-match로 검증
    fc-match 가 요청한 폰트가 없으면 기본 폰트로 폴백하므로, 응답에 폰트명이 포함되어야 설치된 것
    Returns: True(설치됨) / False(미설치=폴백)
    Raises: RuntimeError (fontconfig 자체 미설치)
    """

    try:
        result = subprocess.run(["fc-match", font_name], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise RuntimeError("fc-match 명령어를 찾을 수 없습니다. fontconfig 패키지를 설치하세요"
                           "sudo apt install fontconfig")
    except subprocess.TimeoutExpired:
        logger.warning(f"fc-match 타임아웃 - 폰트 검증 건너뜀: {font_name}")
        return False

    output = result.stdout.strip()
    is_installed = font_name.lower() in output.lower()
    if not is_installed:
        logger.warning(f"폰트 미설치 또는 폴백 발생: '{font_name}' | fc-match 응답: {output}")
    else:
        logger.info(f"폰트 확인: '{font_name}' 정상 설치됨")
    return is_installed

# --------------------------------------------------------------
# 1. 전사 데이터에서 구간별 단어 추출
# --------------------------------------------------------------
def extract_words_for_range(transcript_data: dict, start_sec: float, end_sec: float) -> list[dict]:
    """
    전사 데이터에서 지정 구간의 단어 목록 추출
    각 단어의 타임스탬프를 구간 시작 기준으로 오프셋 (0초부터 시작)
    -> extract_clip()으로 트리밍된 클립의 0초와 자막 타임스탬프가 동기화됨
    Returns: [{"word": "안녕", "start": 0.0, "end": 0.5}, ...]
    """

    words = []
    for seg in transcript_data.get("segments", []):
        if seg.get("end", 0) <= start_sec or seg.get("start", 0) >= end_sec:
            continue

        for w in seg.get("words", []):
            w_start, w_end = w.get("start", 0), w.get("end", 0)
            if w_start >= start_sec and w_end <= end_sec:
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(w_start - start_sec, 3),
                    "end": round(w_end - start_sec, 3),
                })

    logger.info(f"구간 단어 추출: {start_sec:.1f}-{end_sec:.1f}s | {len(words)}개")
    return words

# --------------------------------------------------------------
# 2. ASS 헤더 생성 - [Script Info] + [V4+ Styles]
# --------------------------------------------------------------
def build_ass_header(width: int = 1080, height: int = 1920) -> str:
    """
    Default(일반) + Highlight(강조) 두 스타일을 포함함 ASS 헤더 반환
    17일차: OUTLINE_WIDTH, SHADOW_DEPTH 상수 적용
    """

    return (
        f"[Script Info]\nTitle: YT Shorts AI Subtitles\n"
        f"ScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\n"
        f"WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        f"OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        f"ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{CLR_PRIMARY},&H000000FF,"
        f"{CLR_OUTLINE},{CLR_SHADOW},0,0,0,0,100,100,0,0,1,"
        f"{OUTLINE_WIDTH},{SHADOW_DEPTH},2,20,20,{MARGIN_V},1\n"
        f"Style: Highlight,{FONT_NAME},{FONT_SIZE_HL},{CLR_HIGHLIGHT},&H000000FF,"
        f"{CLR_OUTLINE},{CLR_SHADOW},-1,0,0,0,100,100,0,0,1,"
        f"{OUTLINE_WIDTH},{SHADOW_DEPTH},2,20,20,{MARGIN_V},1\n\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

# --------------------------------------------------------------
# 3. ASS 이벤트 생성 - 단어 단위 카라오케 강조
# --------------------------------------------------------------
def build_ass_events(words: list[dict]) -> str:
    """
    단어 목록을 ASS Dialogue 이벤트로 변환
    WORDS_PER_GROUP개씩 묶어 표시, 현재 발화 단어를 노란색+110% 크기로 강조
    """

    if not words:
        return ""
    
    events = []
    groups = [words[i:i + WORDS_PER_GROUP] for i in range(0, len(words), WORDS_PER_GROUP)]

    for group in groups:
        group_end = group[-1]["end"]
        for idx, active in enumerate(group):
            w_start = active["start"]
            w_end = group[idx + 1]["start"] if idx + 1 < len(group) else group_end

            parts = []
            for j, w in enumerate(group):
                if j == idx:
                    parts.append(
                        f"{{\\c{CLR_HIGHLIGHT}\\fscx110\\fscy110}}{w['word']}{{\\r}}"
                    )
                else:
                    parts.append(w["word"])

            events.append(
                f"Dialogue: 0,{_sec_to_ass(w_start)},{_sec_to_ass(w_end)},"
                f"Default,,0,0,0,,{' '.join(parts)}"
            )

    logger.info(f"ASS 이벤트 생성: {len(events)}개 라인")
    return "\n".join(events)

# --------------------------------------------------------------
# 4. ASS 파일 작성
# --------------------------------------------------------------
def write_ass_file(ass_path: str, header: str, events: str) -> bool:
    """
    ASS 헤더 + 이벤트를 파일로 작성, 성공 여부 반환
    """

    try:
        Path(ass_path).parent.mkdir(parents=True, exist_ok=True)
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(events)
            f.write("\n")
        
        logger.info(f"ASS 파일 생성: {ass_path}")
        return True
    except Exception as e:
        logger.error(f"ASS 파일 작성 실패: {e}")
        return False

# --------------------------------------------------------------
# 5. FFmpeg 자막 합성 - ASS 하드 서브
# --------------------------------------------------------------
async def run_ffmpeg_subtitle(video_path: str, ass_path: str, output_path: str) -> bool:
    """
    FFmpeg ass 필터로 자막을 영상 프레임에 직접 렌더링 (하드 서브)
    주의: ass 필터(libass)는 CPU 전용이므로 -hwaccel을 사용하지 않음
    입력이 60초 짧은 클립이라 CPU 디코딩으로 충분
    """

    # FFmpeg ass 필터 경로 이스케이프: \, :, ' 문자를 백슬래시로 이스케이프'
    escaped_ass = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path, "-vf", f"ass={escaped_ass}",
        "-c:v", "h264_nvenc", "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ), "-c:a", "copy", output_path,
    ]

    logger.info(f"FFmpeg 자막 합성 | {Path(video_path).name} + {Path(ass_path).name}")

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"FFmpeg 자막 합성 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"자막 합성 완료: {output_path}")
    return True

# --------------------------------------------------------------
# 6. FFmpeg 최종 인코딩 - NVENC + 오디오 노멀라이즈 + 속도 조절
# --------------------------------------------------------------
async def run_ffmpeg_encode(video_path: str, output_path: str, speed: float = 1.05, target_lufs: int = -14) -> bool:
    """
    최종 인코딩: setpts(속도) + loudnorm(-14 LUFS) + atempo(오디오 동기화)
    + h264_nvenc(GPU 인코딩) + movflags +faststart(웹 최적화)
    """

    pts_ratio = round(1.0 / speed, 4)
    cmd = [
        "ffmpeg", "-y", "-hwaccel", settings.FFMPEG_HWACCEL,
        "-i", video_path, 
        "-vf", f"setpts={pts_ratio}*PTS",
        "-af", f"atempo={speed},loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-c:v", "h264_nvenc", "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ), 
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path,
    ]

    logger.info(f"최종 인코딩 | 속도: {speed}x | LUFS: {target_lufs} | {Path(output_path).name}")

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"최종 인코딩 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"최종 인코딩 완료: {output_path}")
    return True

# --------------------------------------------------------------
# 유틸리티: 초 -> ASS 시간 형식 (H:MM:SS.CC)
# --------------------------------------------------------------
def _sec_to_ass(seconds: float) -> str:
    """
    초를 ASS 시간 형식으로 변환.
    ex> 65.5초 -> '0:01:05.50'
    """

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"