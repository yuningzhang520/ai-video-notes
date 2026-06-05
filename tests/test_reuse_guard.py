"""Step 6 vision-reuse guard tests (pure; no Anthropic, no images).

Covers match_reusable_block (tight pHash + dt gate, closest-wins) and remap_reused_block
(content reused, every frame-derived field repointed at the new rep, no stale old frameRef).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_visual_blocks as bvb  # noqa: E402

BASE = "0000000000000000"
TWO = "0000000000000003"   # 2 bits from BASE
THREE = "0000000000000007"  # 3 bits from BASE

OLD_BLOCKS = [{"frameRef": "f1", "timestampSec": 100.0, "type": "code",
               "content": "print(1)", "confidence": "high", "model": "m"}]
OLD_PHASH = {"f1": BASE}


def _cand(phash=BASE, ts=101.0):
    return {"id": "frame-0005", "phash": phash, "timestampSec": ts,
            "imagePath": "/static/demo/GDm_uH6VxPY/frames/frame-0005.jpg",
            "suggestedVerifyTimestampSec": 98.0, "ocrText": "new ocr"}


def test_match_reuses_when_near_identical_and_close_in_time():
    assert bvb.match_reusable_block(_cand(BASE, 100.0), OLD_BLOCKS, OLD_PHASH)["frameRef"] == "f1"
    assert bvb.match_reusable_block(_cand(TWO, 108.0), OLD_BLOCKS, OLD_PHASH)["frameRef"] == "f1"  # d=2, dt=8


def test_match_rejects_when_phash_too_far_or_time_too_far():
    assert bvb.match_reusable_block(_cand(THREE, 100.0), OLD_BLOCKS, OLD_PHASH) is None   # d=3 > 2
    assert bvb.match_reusable_block(_cand(BASE, 109.0), OLD_BLOCKS, OLD_PHASH) is None    # dt=9 > 8


def test_match_picks_the_closest_block():
    blocks = [{"frameRef": "far", "timestampSec": 102.0, "type": "code", "content": "a",
               "confidence": "high"},
              {"frameRef": "near", "timestampSec": 101.0, "type": "code", "content": "b",
               "confidence": "high"}]
    phash = {"far": BASE, "near": BASE}
    # both pHash 0; the smaller |dt| (near @101 vs cand @101) wins.
    assert bvb.match_reusable_block(_cand(BASE, 101.0), blocks, phash)["frameRef"] == "near"


def test_remap_reuses_content_but_repoints_every_frame_field():
    out = bvb.remap_reused_block(OLD_BLOCKS[0], _cand())
    assert out["type"] == "code" and out["content"] == "print(1)" and out["confidence"] == "high"
    # every frame-derived field points at the NEW rep; provenance recorded; cost zeroed.
    assert out["frameRef"] == "frame-0005"
    assert out["sourceImagePath"].endswith("frame-0005.jpg")
    assert out["timestampSec"] == 101.0 and out["suggestedVerifyTimestampSec"] == 98.0
    assert out["ocrText"] == "new ocr"
    assert out["reusedFromFrameRef"] == "f1"
    assert out["inputTokens"] == 0 and out["outputTokens"] == 0 and out["error"] is None
    # no stale old ref anywhere in the remapped entry.
    assert "f1" not in {out["frameRef"], out["sourceImagePath"]}
