# 계층: 테스트 계층 (unit test)
# 역할: transcript_chunker 모듈의 청크 분할 + 병합/재랭킹 로직 단위 테스트
# 의존: pytest, app.services.transcript_chunker
# MVA 원칙: 순수 함수 테스트 - 외부 의존성(LLM, DB) 없음

"""transcript_chunker 단위 테스트 - 청크 분할(4) + 재랭킹(5) 케이스"""

import pytest
from app.services.transcript_chunker import (
    split_transcript_into_chunks, merge_and_rerank_highlights
)

# --------------------------------------------------------------
# 테스트 픽스처 (헬퍼 함수)
# --------------------------------------------------------------

def _make_segment(start: float, end: float, text: str = "발화") -> dict:
    """전사 세그먼트 생성 헬퍼"""
    
    return {
        "id": 0, "start": start, "end": end, "text": text,
        "words": [{"word": text, "start": start, "end": end, "probability": 0.9}],
    }

def _make_transcript(duration_sec: float, segments: list[dict]) -> dict:
    """전사 dict 생성 헬퍼"""
    
    return {
        "language": "ko", "language_probability": 0.98,
        "duration_sec": duration_sec, "segments": segments,
    }

def _make_highlight(start: float, end: float, score: float) -> dict:
    """하이라이트 dict 생성 헬퍼"""
    
    return {
        "start_sec": start, "end_sec": end, "hook_score": score,
        "reason": "test", "title_suggestion": "test", "tags": [],
    }

# --------------------------------------------------------------
# 청크 분할 테스트 (4개)
# --------------------------------------------------------------
class TestSplitTranscriptIntoChunks:
    """split_transcript_into_chunks 테스트"""

    def test_short_video_single_chunk(self):
        """CASE 1: 짧은 영상 (600초 이하)은 청크 1개, 오버랩 없음"""

        segments = [_make_segment(0, 10), _make_segment(10, 20)]
        transcript = _make_transcript(duration_sec=300.0, segments=segments)

        chunks = split_transcript_into_chunks(transcript, chunk_duration_sec=600.0, overlap_sec=30.0)

        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["start_offset_sec"] == 0.0
        assert chunks[0]["end_offset_sec"] == 300.0
        assert chunks[0]["language"] == "ko"
        assert len(chunks[0]["segments"]) == 2

    def test_medium_video_multiple_chunks_with_overlap(self):
        """CASE 2: 중간 영상 (1500초)은 청크 3개, 오버랩 정상"""

        # 청크1: [0, 615], 청크2: [585, 1215], 청크3: [1185, 1500]
        segments = [_make_segment(i * 60, (i + 1) * 60) for i in range(25)]
        transcript = _make_transcript(duration_sec=1500.0, segments=segments)
        chunks = split_transcript_into_chunks(transcript, chunk_duration_sec=600.0, overlap_sec=30.0)

        assert len(chunks) == 3
        assert chunks[0]["start_offset_sec"] == 0.0
        assert chunks[0]["end_offset_sec"] == 615.0
        assert chunks[1]["start_offset_sec"] == 585.0
        assert chunks[1]["end_offset_sec"] == 1215.0
        assert chunks[2]["start_offset_sec"] == 1185.0
        assert chunks[2]["end_offset_sec"] == 1500.0

    def test_long_video(self):
        """CASE 3: 장편 영상 (93분, 16일차 실패 케이스)은 청크 10개"""

        segments = [_make_segment(i * 3.0, (i + 1) * 3.0) for i in range(1862)]
        transcript = _make_transcript(duration_sec=5587.0, segments=segments)
        chunks = split_transcript_into_chunks(transcript, chunk_duration_sec=600.0, overlap_sec=30.0)

        # 5587 / 600 = 9.3 -> 10개 (마지막 청크 잔여 187초)
        assert len(chunks) == 10
        # 마지막 청크는 total_duration까지
        assert chunks[-1]["end_offset_sec"] == 5587.0
        # 순차적으로 인덱스 부여
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_empty_transcript(self):
        """CASE 4: 빈 전사 데이터 -> 청크 1개(빈 segment)"""

        transcript = _make_transcript(duration_sec=0.0, segments=[])
        chunks = split_transcript_into_chunks(transcript, chunk_duration_sec=600.0, overlap_sec=30.0)

        assert len(chunks) == 1
        assert chunks[0]["segments"] == []
        assert chunks[0]["duration_sec"] == 0.0

# --------------------------------------------------------------
# 재랭킹 테스트 (5개)
# --------------------------------------------------------------
class TestMergeAndRerankHighlights:
    """merge_and_rerank_highlights 테스트"""

    def test_non_overlapping_candidates_all_kept(self):
        """CASE 5: 서로 겹치지 않는 후보들은 모두 유지 + hook_score 내림차순"""

        chunk_results = [
            [_make_highlight(10, 30, 0.9), _make_highlight(100, 120, 0.85)],
            [_make_highlight(700, 720, 0.8), _make_highlight(800, 820, 0.75)],
        ]

        result = merge_and_rerank_highlights(chunk_results, max_shorts=5, iou_threshold=0.3)

        assert len(result) == 4
        scores = [h["hook_score"] for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_iou_above_threshold_keeps_highler_score(self):
        """CASE 6: IoU 0.333 겹침 -> 높은 점수만 유지 (A:[10,30], B[20,40] = IoU 0.333)"""

        chunk_results = [
            [_make_highlight(10, 30, 0.9)],
            [_make_highlight(20, 40, 0.7)],
        ]

        result = merge_and_rerank_highlights(chunk_results, max_shorts=5, iou_threshold=0.3)

        assert len(result) == 1
        assert result[0]["hook_score"] == 0.9
        assert result[0]["start_sec"] == 10

    def test_iou_below_threshold_keeps_both(self):
        """CASE 7: IoU 0.053 < 0.3 임계값 -> 둘다 유지"""

        chunk_results = [
            [_make_highlight(10, 30, 0.9)],
            [_make_highlight(28, 48, 0.7)],
        ]

        result = merge_and_rerank_highlights(chunk_results, max_shorts=5, iou_threshold=0.3)

        assert len(result) == 2

    def test_max_shorts_limit_applied(self):
        """CASE 8: max_shorts 초과 시 상위 점수만 선정"""

        chunk_results = [
            [_make_highlight(0, 20, 0.9), 
             _make_highlight(100, 120, 0.85),
             _make_highlight(200, 220, 0.7),
             _make_highlight(300, 320, 0.6),
             _make_highlight(400, 420, 0.5),],
        ]

        result = merge_and_rerank_highlights(chunk_results, max_shorts=3, iou_threshold=0.3)

        assert len(result) == 3
        assert [h["hook_score"] for h in result] == [0.9, 0.85, 0.7]

    def test_empty_chunk_results_returns_empty(self):
        """CASE 9: 모든 청크가 빈 경우 빈 리스트"""

        result = merge_and_rerank_highlights([[], [], []], max_shorts=5, iou_threshold=0.3)
        assert result == []