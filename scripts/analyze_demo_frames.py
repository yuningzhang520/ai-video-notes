"""Step 4A: the deterministic text GATE over the committed change-aware representatives.

Step 6 moved frame SELECTION (dense -> cropped-pHash segment -> OCR-superset representative) into
``scripts/create_demo_artifacts.py``, which commits the representatives + their per-frame signals
(phash / ocrText / textScore / segment provenance) in ``manifest.json``. This step is what's left
of 4A: apply the on-screen-text gate to each representative and emit the committed artifact

    data/demo/GDm_uH6VxPY/frame_analysis.json

that Step 4B reads to know WHICH representatives are worth a vision call (and where to seek on
verify). NO LLM, NO network, NO re-OCR -- it reads the signals the selector already computed.
Every committed frame IS a representative (selection already deduped); the only decision here is
the gate: textScore >= MIN_TEXT_SCORE.

Run from the repo root:  python scripts/analyze_demo_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.frame_select import MIN_TEXT_SCORE  # noqa: E402

VIDEO_ID = "GDm_uH6VxPY"
DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANALYSIS_PATH = DATA_DIR / "frame_analysis.json"


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def gate(text_score: int) -> tuple[bool, str]:
    if text_score >= MIN_TEXT_SCORE:
        return True, (f"representative with {text_score} confident on-screen words "
                      f"(>= {MIN_TEXT_SCORE}); vision candidate")
    return False, (f"representative but only {text_score} confident words "
                   f"(< {MIN_TEXT_SCORE}); below text gate, skipped")


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("missing manifest -- run scripts/create_demo_artifacts.py first")
    manifest = json.loads(MANIFEST_PATH.read_text())

    frames = []
    for f in manifest["frames"]:
        if not (REPO_ROOT / f["imagePath"].lstrip("/")).is_file():
            raise SystemExit(f"frame_analysis references missing file: {f['imagePath']}")
        is_cand, reason = gate(int(f["textScore"]))
        frames.append({
            "id": f["id"], "timestampSec": f["timestampSec"], "imagePath": f["imagePath"],
            "phash": f["phash"], "segmentId": f["segmentId"],
            "selectionReason": f["selectionReason"], "segMemberTimestamps": f["segMemberTimestamps"],
            "isRepresentative": True,          # every committed frame is a selected representative
            "ocrText": f["ocrText"], "wordCount": f["wordCount"], "textLength": f["textLength"],
            "textScore": f["textScore"], "ocrConfidence": f["ocrConfidence"],
            "isVisionCandidate": is_cand, "candidateReason": reason,
            "suggestedVerifyTimestampSec": max(float(f["timestampSec"]) - 3.0, 0.0),
        })

    candidates = [f for f in frames if f["isVisionCandidate"]]
    text_positive = [f for f in frames if f["textScore"] > 0]
    out = {
        "videoId": VIDEO_ID, "source": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "params": {
            "minTextScore": MIN_TEXT_SCORE,
            "segThreshold": manifest["selection"]["segThreshold"],
            "slideRegionFrac": manifest["selection"]["slideRegionFrac"],
            "adjacentOverlap": manifest["selection"]["adjacentOverlap"],
        },
        "selection": manifest["selection"],
        "counts": {
            "totalFrames": len(frames), "representatives": len(frames),
            "textPositiveFrames": len(text_positive), "visionCandidates": len(candidates),
        },
        "frames": frames,
    }
    ANALYSIS_PATH.write_text(json.dumps(out, indent=2) + "\n")

    print("=== Step 4A text gate ===")
    print(f"  representatives:  {len(frames)}")
    print(f"  text-positive:    {len(text_positive)}")
    print(f"  vision candidates:{len(candidates)}")
    print(f"  wrote:            {ANALYSIS_PATH.relative_to(REPO_ROOT)}")
    print(f"\n  candidate ids: {[f['id'] for f in candidates]}")


if __name__ == "__main__":
    main()
