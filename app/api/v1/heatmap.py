# 계층: API 계층 (Controller)
# 역할: 히트맵 수집 HTTP 엔드포인트 - 프론트 엔드에서 데이터 수집 UI 제공
# 의존: HeatmapCollector (서비스 계층)
# MVA 원칙: 비즈니스 로직 없이 서비스에 위임
# 20일차 신규: 파인튜닝 데이터 수집을 프론트엔드에서 실행 가능하게 함
#
# 엔드포인트:
#   POST /api/v1/heatmap/collect            단일 URL 히트맵 수집
#   POST /api/v1/heatmap/collect-batch      다수 URL 일괄 수집
#   POST /api/v1/heatmap/channel-urls       채널/재생목록에서 URL 목록 추출
#   GET /api/v1/heatmap/results             수집 결과 JSONL 목록 조회
#   GET /api/v1/heatmap/results/{filename}  특정 JSONL 파일 내용 조회

"""
히트맵 수집 엔드포인트 - YouTube "Most Replayed" 히트맵 데이터 수집 API
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.services.heatmap_collector import HeatmapCollector

router = APIRouter()

# --------------------------------------------------------------
# 요청/응답 스키마
# --------------------------------------------------------------

class CollectRequest(BaseModel):
    url: str

class BatchCollectRequest(BaseModel):
    urls: list[str]
    rate_limit_sec: float = 2.0

class ChannelUrlRequest(BaseModel):
    channel_url: str
    max_videos: int = 50

class CollectResult(BaseModel):
    video_id: str
    title: str
    duration_sec: float
    heatmap_count: int
    peak_count: int
    status: str

class BatchResult(BaseModel):
    success: int
    skipped: int
    failed: int
    total: int
    filename: str
    results: list[CollectResult]

class ChannelUrlResult(BaseModel):
    urls: list[str]
    total: int
    channel_url: str

# --------------------------------------------------------------
# 엔트 포인트
# --------------------------------------------------------------

@router.post("/collect", response_model=CollectResult)
async def collect_single(req: CollectRequest):
    """단일 URL 히트맵 수집"""

    collector = HeatmapCollector()
    result = await collector.collect_single(req.url)

    if not result:
        return CollectResult(
            video_id="", title="", duration_sec=0,
            heatmap_count=0, peak_count=0, status="skipped",
        )
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"heatmaps_{today}.jsonl"
    collector.append_to_jsonl(result, filename)

    return CollectResult(
        video_id=result["video_id"],
        title=result["title"],
        duration_sec=result["duration_sec"],
        heatmap_count=len(result["heatmap"]),
        peak_count=len(result["peak_segments"]),
        status="collected",
    )

@router.post("/collect-batch", response_model=BatchResult)
async def collect_batch(req: BatchCollectRequest):
    """다수 URL 일괄 히트맵 수집"""

    # 중복 제거
    seen = set()
    urls = [u for u in req.urls if u not in seen and not seen.add(u)]

    if not urls:
        raise HTTPException(status_code=400, detail="URL 목록이 비어 있습니다.")
    
    collector = HeatmapCollector()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"heatmaps_{today}.jsonl"

    stats = {"success": 0, "skipped": 0, "failed": 0}
    results = []

    for idx, url in enumerate(urls):
        logger.info(f"[배치 {idx+1}/{len(urls)}] {url}")
        try:
            result = await collector.collect_single(url)
            if result:
                collector.append_to_jsonl(result, filename)
                stats["success"] += 1
                results.append(CollectResult(
                    video_id=result["video_id"], title=result["title"],
                    duration_sec=result["duration_sec"],
                    heatmap_count=len(result["heatmap"]),
                    peak_count=len(result["peak_segments"]),
                    status="collected",
                ))
            else:
                stats["skipped"] += 1
                results.append(CollectResult(
                    video_id="", title="", duration_sec=0,
                    heatmap_count=0, peak_count=0, status="skipped",
                ))
        except Exception as e:
            stats["skipped"] += 1
            results.append(CollectResult(
                video_id="", title=str(e), duration_sec=0,
                heatmap_count=0, peak_count=0, status="failed"
            ))

        if idx < len(urls) - 1:
            await asyncio.sleep(req.rate_limit_sec)

    return BatchResult(**stats, total=len(urls), filename=filename, results=results)

@router.post("/channel-urls", response_model=ChannelUrlResult)
async def extract_channel_urls(req: ChannelUrlRequest):
    """채널/재생목록 URL에서 개별 영상 URL 목록 추출"""

    cmd = [
        "yt-dlp", "--flat-playlist", "--print", "url",
        "--no-warnings", "--playlist-end", str(req.max_videos),
        req.channel_url,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"yt-dlp 실패: {stderr.decode(errors='replace')[:200]}",
            )
        
        urls = [
            line.strip() for line in stdout.decode().splitlines()
            if line.strip()
        ]

        return ChannelUrlResult(urls=urls, total=len(urls), channel_url=req.channel_url)
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="채널 URL 추출 타임아웃")

@router.get("/results")
async def list_results():
    """수집된 JSONL 파일 목록 조회"""

    output_dir = settings.heatmap_output_path
    files = sorted(output_dir.glob("*.jsonl"), reverse=True)

    result = []
    for f in files:
        line_count = sum(1 for _ in open(f, encoding="utf-8"))
        result.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "video_count": line_count,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })

    return {"files": result, "total": len(result)}

@router.get("/results/{filename}")
async def get_result_detail(filename: str):
    """특정 JSONL 파일 내용 조회"""

    filepath = settings.heatmap_output_path / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"파일 없음: {filename}")
    
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                entries.append({
                    "video_id": data.get("video_id"),
                    "title": data.get("title"),
                    "duration_sec": data.get("duration_sec"),
                    "heatmap_count": len(data.get("heatmap", [])),
                    "peak_count": len(data.get("peak_segments", []))
                })

    return {"filename": filename, "entries": entries, "total": len(entries)}