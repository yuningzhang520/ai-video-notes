"""Frame-selection tests (Tesseract-free; the binary is only needed at build time).

Two layers:
  - the PURE ``app.frame_select.phash_distance`` Hamming rule (synthetic hex hashes, no
    images, no OCR engine);
  - the committed ``frame_analysis.json`` CONTRACT that Step 4B consumes -- counts in range,
    every reference resolves, and the change-aware selection / gate / verify-lead invariants
    hold (every committed frame is an already-selected representative with segment provenance).

The change-aware selection PRIMITIVES (segment / choose_representative / collapse_adjacent_reps)
are covered in test_frame_select_step6.py.
"""

import json
from pathlib import Path

from app.frame_select import phash_distance

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_PATH = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY" / "frame_analysis.json"

ZERO = "0000000000000000"
TWO_BITS = "0000000000000003"   # Hamming distance 2 from ZERO
ALL_ONES = "ffffffffffffffff"   # Hamming distance 64 from ZERO


def test_phash_distance_is_xor_popcount():
    assert phash_distance(ZERO, ZERO) == 0
    assert phash_distance(ZERO, TWO_BITS) == 2
    assert phash_distance(ZERO, ALL_ONES) == 64


# --- Committed artifact contract (what Step 4B reads) --------------------------------

def test_committed_frame_analysis_contract():
    # Step 6 schema: every committed frame is an already-selected representative (the dense set
    # was segmented + collapsed upstream), so the only per-frame decision here is the text gate.
    data = json.loads(ANALYSIS_PATH.read_text())
    frames = data["frames"]

    assert data["counts"]["totalFrames"] == len(frames)
    assert data["counts"]["representatives"] == len(frames)

    candidates = [f for f in frames if f["isVisionCandidate"]]
    assert data["counts"]["visionCandidates"] == len(candidates)
    # Change-aware selection over a demo-dense video lands a larger, bounded candidate set.
    assert 25 <= len(candidates) <= 50, f"{len(candidates)} candidates is outside 25-50"

    for f in frames:
        # Every referenced frame image actually exists on disk.
        assert (REPO_ROOT / f["imagePath"].lstrip("/")).is_file(), f["imagePath"]
        # Selection already deduped: each committed frame is a representative with its provenance.
        assert f["isRepresentative"] is True
        assert isinstance(f["segmentId"], int) and f["selectionReason"]
        # Candidates are exactly representatives that clear the text gate.
        if f["isVisionCandidate"]:
            assert f["textScore"] >= data["params"]["minTextScore"]
        # Verify-seek lead is the clamped lead defined in the spec.
        assert f["suggestedVerifyTimestampSec"] == max(f["timestampSec"] - 3, 0)
