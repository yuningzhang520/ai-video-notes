"""Step 4B: visual-block tests. NONE of these call Anthropic.

Two layers:
  - the PURE ``classify_result`` success/failure rule, exercised with synthetic
    ``VisionResult`` objects (a valid block, plus each failure mode -> a FAILED entry);
  - the committed ``visual_blocks.json`` CONTRACT that Step 4C will consume -- one entry
    per 4A candidate, at most one block per frame, every reference resolves, and the
    verify-lead / counts invariants hold (failed entries, if any, parse explicitly).
"""

import json
import sys
from pathlib import Path

from app.vision import VALID_CONFIDENCE, VALID_TYPES, VisionResult

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_visual_blocks as bvb  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY"
BLOCKS_PATH = DATA_DIR / "visual_blocks.json"
ANALYSIS_PATH = DATA_DIR / "frame_analysis.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

CANDIDATE = {
    "id": "frame-0099",
    "timestampSec": 100.0,
    "suggestedVerifyTimestampSec": 97.0,
    "imagePath": "/static/demo/GDm_uH6VxPY/frames/frame-0099.jpg",
    "ocrText": "noisy ocr",
    "candidateReason": "representative with 20 confident on-screen words (>= 8); vision candidate",
}


def _ok_result(**over):
    base = dict(ok=True, type="code", content="print(1)", confidence="high",
                model="claude-sonnet-4-6")
    base.update(over)
    return VisionResult(**base)


# --- Pure classify_result rule -------------------------------------------------------

def test_classify_success_is_one_block_with_no_error():
    entry = bvb.classify_result(CANDIDATE, _ok_result())
    assert entry["error"] is None
    assert bvb.is_success(entry)
    assert entry["type"] == "code" and entry["confidence"] == "high"
    # Carries the 4A provenance fields verbatim.
    assert entry["frameRef"] == "frame-0099"
    assert entry["timestampSec"] == 100.0
    assert entry["suggestedVerifyTimestampSec"] == 97.0
    assert entry["sourceImagePath"] == CANDIDATE["imagePath"]
    assert entry["ocrText"] == "noisy ocr"


def test_classify_failure_modes_are_failed_entries_not_blocks():
    # Each bad result yields exactly ONE entry, marked failed (error set), never a success.
    cases = {
        "api error":          VisionResult(ok=False, error="RateLimitError: slow down",
                                           model="claude-sonnet-4-6"),
        "empty content":      _ok_result(content="   "),
        "invalid type":       _ok_result(type="screenshot"),
        "invalid confidence": _ok_result(confidence="definitely"),
    }
    for label, result in cases.items():
        entry = bvb.classify_result(CANDIDATE, result)
        assert not bvb.is_success(entry), label
        assert entry["error"], label


# --- Committed artifact contract -----------------------------------------------------

def test_committed_visual_blocks_contract():
    data = json.loads(BLOCKS_PATH.read_text())
    analysis = json.loads(ANALYSIS_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())

    blocks = data["blocks"]
    manifest_ids = {f["id"] for f in manifest["frames"]}
    candidate_ids = {f["id"] for f in analysis["frames"]
                     if f["isVisionCandidate"] and f["isRepresentative"]}

    # Only the 4A candidates were processed -- one entry each, never a non-candidate.
    assert 25 <= len(candidate_ids) <= 50   # change-aware selection over a demo-dense video
    assert data["counts"]["candidates"] == len(candidate_ids)
    assert {b["frameRef"] for b in blocks} == candidate_ids

    # At most ONE block per frame.
    refs = [b["frameRef"] for b in blocks]
    assert len(refs) == len(set(refs))

    # Counts are internally consistent.
    successes = [b for b in blocks if bvb.is_success(b)]
    failures = [b for b in blocks if not bvb.is_success(b)]
    assert data["counts"]["successfulBlocks"] == len(successes)
    assert data["counts"]["failedBlocks"] == len(failures)
    assert len(successes) + len(failures) == len(blocks)

    required = {"frameRef", "timestampSec", "suggestedVerifyTimestampSec", "type",
                "content", "confidence", "sourceImagePath", "ocrText", "model", "error"}
    for b in blocks:
        assert required <= set(b), f"{b['frameRef']} missing {required - set(b)}"
        # Verify-lead is a clamped lead: 0 <= suggested <= timestamp.
        assert 0 <= b["suggestedVerifyTimestampSec"] <= b["timestampSec"]
        # Every block resolves to a real candidate frame and its image exists on disk.
        assert b["frameRef"] in candidate_ids and b["frameRef"] in manifest_ids
        assert (REPO_ROOT / b["sourceImagePath"].lstrip("/")).is_file()

        if bvb.is_success(b):
            assert b["error"] is None
            assert b["type"] in VALID_TYPES
            assert b["confidence"] in VALID_CONFIDENCE
            assert b["content"] and b["content"].strip()
        else:
            # A failed entry is represented explicitly: it has an error and no usable block.
            assert b["error"]

    # Vision-reuse (Step 6): reused blocks carry provenance and a NEW (remapped) frameRef -- the
    # reused content is repointed at the new representative, so no stale old id leaks into frameRef.
    reused = [b for b in blocks if b.get("reusedFromFrameRef")]
    assert reused, "expected some reused blocks after a re-sample"
    for b in reused:
        assert b["frameRef"] in candidate_ids and b["frameRef"] != b["reusedFromFrameRef"]
