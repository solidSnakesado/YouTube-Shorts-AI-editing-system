# 계층: 비즈니스 로직 계층 (Service 헬퍼)
# 역할: FFmpeg 클립 추출,  YOLOv8 피사체 추적, 카메라 스무딩, 
#       적응형 리프레이밍 전략, FFmpeg 크롭 실행, (editing_service.py 300줄 규칙으로 분리)
# 의존: 없음 (gpu_manager가 반환한 YOLO 모델을 인자로 받아 사용)
# MVA 원칙: 인프라 책임(모델 로드/언로드)은 gpu_manager에 위임
# 흐름: extract_clip -> detect_subjects -> smooth_trajectory -> choose_strategy
#      -> build_crop_timeline -> run_ffmpeg_reframe
# 8~10일차 신규 / 11~12일차 수정: extract_clip() 추가
# 21일차 변경:
#   - ASPECT_RATIOS에 16:10 추가, TARGET_RESOLUTIONS 딕셔너리 신규
#   - run_ffmpeg_reframe()에서 aspect_ratio로 target 해상도 자동 계산
#   - run_ffmpeg_resize() 신규: 기존 영상을 다른 비율로 리사이징

"""
리프레이밍 엔진 -클립 추출, 피사체 추적, 스무딩, 적응형 크롭, FFmpeg 실행
16:9 가로 영상을 9:16 세로 쇼츠로 변환하는 핵심 로직
""" 

import asyncio
import math
from pathlib import Path
from typing import Any

from loguru import logger
from app.core.config import settings

# --- 상수 ---
ASPECT_RATIOS           = {
    "9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1), 
    "4:5": (4, 5), "4:3": (4, 3), "16:10": (16, 10)
}

# 비율별 기본 출력 해상도 (짧은 변 1080px 기준)
TARGET_RESOLUTIONS = {
    "9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), 
    "4:5": (1080, 1350), "4:3": (1440, 1080), "16:10": (1728, 1080),
}

PERSON_CLASS_ID         = 0                     # COCO 데이터셋 person 클래스
STRATEGY_STATIC         = "static"              # 고정 모드
STRATEGY_PAN            = "pan"                 # 팬 모드
STRATEGY_TRACK          = "track"               # 추적 모드
STRATEGY_LETTERBOX      = "letterbox"           # 레터박스 모드
SMOOTHING_ALPHA         = 0.15                  # EMA 스무딩 계수 (0에 가까울수록 부드러움)
MIN_MOVE_THRESHOLD      = 5                     # 이 픽셀 이하 이동은 무시 (떨림 방지)
SCENE_CUT_DISTANCE      = 200                   # 장면 전환 판단 임계값 (픽셀)
YOLO_CONF_THRESHOLD     = 0.4                   # YOLO 감지 신뢰도 임계값
YOLO_SAMPLE_FPS         = 5                     # 초당 샘플링 프레임 수

# --------------------------------------------------------------
# 0. 클립 추출 - 소스에서 start_sec - end_sec 구간만 추출 (11~12일차 신규)
# --------------------------------------------------------------
async def extract_clip(source_path: str, clip_path: str, start_sec: float, end_sec: float) -> bool:
    """FFmpeg로 소스 영상에서 지정 구간만 추출 (-ss 입력 시크, -c copy 초고속)"""

    duration = round(end_sec - start_sec, 3)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(round(start_sec, 3)),
        "-i", source_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        clip_path,
    ]
    logger.info(f"클립 추출 | {start_sec:.1f}-{end_sec:.1f}s ({duration:.1f}s) | "
                f"{Path(source_path).name} -> {Path(clip_path).name}")
    
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"클립 추출 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"클립 추출 완료: {clip_path}")
    return True
    

# --------------------------------------------------------------
# 1. 피사체 탐지 - YOLOv8로 프레임별 인물 위치 추적
# --------------------------------------------------------------
def detect_subjects(yolo_model: Any, video_path: str, sample_fps: int = YOLO_SAMPLE_FPS) -> list[dict]:
    """YOLOv8로 프레임별 피사체(인물) 위치 탐지, 면적 최대 박스 = 주 피사체"""

    logger.info(f"피사체 탐지 시작 | 영상: {Path(video_path).name} | {sample_fps}fps")

    results = yolo_model.predict(
        source=video_path, stream=True, conf=YOLO_CONF_THRESHOLD,
        classes=[PERSON_CLASS_ID], verbose=False,
        vid_stride=max(1, 30 // sample_fps),
    )

    detections = []
    for frame_idx, result in enumerate(results):
        orig_h, orig_w = result.orig_shape
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            # 면적이 가장 큰 박스 = 주 피사체
            areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
            best = areas.argmax()
            cx, cy, w, h = boxes.xywh[best].cpu().numpy().tolist()
        else:
            cx, cy, w, h = orig_w / 2, orig_h / 2, 0, 0

        detections.append({
            "frame_idx": frame_idx, "time_sec": round(frame_idx / max(sample_fps, 1), 3),
            "cx": round(cx), "cy": round(cy), "w": round(w), "h": round(h),
            "orig_w": orig_w, "orig_h": orig_h,
        })

    logger.info(f"피사체 탐지 완료 | 프레임: {len(detections)}개")
    return detections

# --------------------------------------------------------------
# 2. 카메라 이동 스무딩 - EMA(Exponential Moveing Average) 필터
# --------------------------------------------------------------
def smooth_trajectory(detections: list[dict], alpha: float = SMOOTHING_ALPHA) -> list[dict]:
    """EMA 저대역 통과 필터로 카메라 이동 경로 스무딩, 장면 전환 시 즉시 점프"""

    if not detections:
        return detections
    
    sx, sy = float(detections[0]["cx"]), float(detections[0]["cy"])
    detections[0]["smooth_cx"] = round(sx)
    detections[0]["smooth_cy"] = round(sy)

    for i in range(1, len(detections)):
        cx, cy = float(detections[i]["cx"]), float(detections[i]["cy"])
        dist = math.hypot(cx - sx, cy - sy)

        if dist > SCENE_CUT_DISTANCE:
            sx, sy = cx, cy                         # 장면 전환 -> 즉시 점프
        elif dist >= MIN_MOVE_THRESHOLD:
            sx = alpha * cx + (1 - alpha) * sx      # EMA 스무딩
            sy = alpha * cy + (1 - alpha) * sy

        detections[i]["smooth_cx"] = round(sx)
        detections[i]["smooth_cy"] = round(sy)

    return detections

# --------------------------------------------------------------
# 3. 적응형 리프레이밍 전략 선택
# --------------------------------------------------------------
def choose_strategy(detections: list[dict]) -> str:
    """피사체 이동 패턴 분석 -> 리프레이밍 전략 결정(static/pan/track/letterbox)"""

    if len(detections) < 2:
        return STRATEGY_STATIC
    
    no_detect = sum(1 for d in detections if d.get("w", 0) == 0)
    if no_detect / len(detections) > 0.5:
        return STRATEGY_LETTERBOX
    
    moves = []
    for i in range(1, len(detections)):
        dx = detections[i].get("smooth_cx", 0) - detections[i - 1].get("smooth_cx", 0)
        dy = detections[i].get("smooth_cy", 0) - detections[i - 1].get("smooth_cy", 0)
        moves.append((dx, dy))

    avg_move = sum(math.hypot(dx, dy) for dx, dy in moves) / len(moves)
    if avg_move < MIN_MOVE_THRESHOLD:
        return STRATEGY_STATIC
    
    x_signs = [1 if dx > 0 else -1 for dx, _ in moves if abs(dx) > 1]
    if x_signs:
        dominant = max(x_signs.count(1), x_signs.count(-1)) / len(x_signs)
        if dominant > 0.8:
            return STRATEGY_PAN
        
    return STRATEGY_TRACK

# --------------------------------------------------------------
# 4. 크롭 타임라인 생성 - 프레임별 크롭 좌표
# --------------------------------------------------------------
def build_crop_timeline(detections: list[dict], strategy: str, aspect_ratio: str = "9:16") -> list[dict]:
    """프레임별 (crop_x, crop_y, crop_w, crop_h) 타임라인 생성"""

    if not detections:
        return []
    
    ar_w, ar_h = ASPECT_RATIOS.get(aspect_ratio, (9, 16))
    orig_w = detections[0].get("orig_w", 1920)
    orig_h = detections[0].get("orig_h", 1080)

    # 높이 기준 크롭 크기, 너비 초과 시 너비 기준 재계산
    crop_h = orig_h
    crop_w = int(crop_h * ar_w / ar_h)
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(crop_w * ar_h / ar_w)

    center_x, center_y = orig_w // 2, orig_h // 2       # 중앙 기준 크롭 (레터박스 영상 대응)
    timeline = []
    for d in detections:
        if strategy == STRATEGY_LETTERBOX:
            cx_pos = center_x - crop_w // 2
            cy_pos = center_y - crop_h // 2
            is_lb = True
        else:
            cx = d.get("smooth_cx", center_x)
            cy = d.get("smooth_cy", center_y)
            cx_pos = max(0, min(cx - crop_w // 2, orig_w - crop_w))
            cy_pos = max(0, min(cy - crop_h // 2, orig_h - crop_h))
            is_lb = False

        timeline.append({
            "time_sec": d["time_sec"], "crop_x": round(cx_pos), "crop_y": round(cy_pos),
            "crop_w": crop_w, "crop_h": crop_h, "letterbox": is_lb,
        })

    logger.info(f"크롭 타임라인 | 전략: {strategy} | 크롭: {crop_w}x{crop_h} | {len(timeline)}프레임")
    return timeline

# --------------------------------------------------------------
# 5. FFmpeg 리프레이밍 실행 - 크롭 + 스케일링
# --------------------------------------------------------------
async def run_ffmpeg_reframe(source_path: str, output_path: str, timeline: list[dict], 
    aspect_ratio: str = "9:16") -> bool:
    """
    FFmpeg로 크롭 타임라인 기반 리프레이밍 실행, 프로토타입 단순화: 타임라인 중앙값 크롭 고정 크롭
    21일차: aspect_ratio에서 target 해상도 자동 결정
    """

    if not timeline:
        logger.error("빈 크롭 타임라인")
        return False
    
    target_width, target_height = TARGET_RESOLUTIONS.get(aspect_ratio, (1080, 1920))
    
    mid = timeline[len(timeline) // 2]
    crop_x, crop_y = mid["crop_x"], mid["crop_y"]
    crop_w, crop_h = mid["crop_w"], mid["crop_h"]

    if mid.get("letterbox"):
        vf = (f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
              f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black")
    else:
        vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_width}:{target_height}"

    cmd = [
        "ffmpeg", "-y", "-hwaccel", settings.FFMPEG_HWACCEL,
        "-i", source_path, "-vf", vf,
        "-c:v", "h264_nvenc", "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ), "-c:a", "copy", output_path,
    ]

    logger.info(f"FFmpeg 리프레이밍 | {crop_w}x{crop_h}+{crop_x}+{crop_y} -> {target_width}x{target_height}")

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"FFmpeg 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"리프레이밍 완료: {output_path}")
    return True

# --------------------------------------------------------------
# 6. 리사이징 - 기존 영상을 다른 비율로 변환 (21일차 신규)
# --------------------------------------------------------------
async def run_ffmpeg_resize(source_path: str, output_path: str, aspect_ratio: str) -> bool:
    """
    기존 영상을 지정 비율로 리사이징 (크롭 + 스케일 또는 레터박스)
    원본 비율과 타겟 비율을 비교하여 타겟이 더 좁으면 중앙 크롭 후 스케일, 타겟이 더 넓으면 레터박스(패딩) 후 스케일
    """

    if aspect_ratio not in TARGET_RESOLUTIONS:
        logger.error(f"미지원 종횡비: {aspect_ratio} (지원: {list(TARGET_RESOLUTIONS.keys())})")
        return False
    
    tw, th = TARGET_RESOLUTIONS[aspect_ratio]
    target_ar = tw / th

    # 레터박스 방식: 원본 비율 유지, 남는 공간은 검정 패딩
    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
          f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black")
    
    cmd = [
        "ffmpeg", "-y", "-hwaccel", settings.FFMPEG_HWACCEL,
        "-i", source_path, "-vf", vf,
        "-c:v", "h264_nvenc", "-preset", settings.NVENC_PRESET,
        "-cq", str(settings.NVENC_CQ), "-c:a", "copy", output_path,
    ]

    logger.info(f"리사이징 | {aspect_ratio} ({tw}x{th}) | {Path(source_path).name}")

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"리사이징 FFmpeg 실패: {stderr.decode()[:500]}")
        return False
    
    logger.info(f"리사이징 완료: {output_path}")
    return True