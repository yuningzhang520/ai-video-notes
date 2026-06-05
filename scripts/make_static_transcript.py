"""Emit a slim, SERVED copy of the transcript for the reader (Step 5B-redesign).

The reader shows "what was said" at a bullet's timestamp. Spoken bullets already carry their
caption span in note_full.json, but VISUAL bullets don't -- and the full transcript lives in
data/ (not served). This writes a slim {start, text} copy under static/ so the client can fetch
it once and snap the span at a visual bullet's timestamp client-side (render-only; no note /
models / structure / grounding change, no vision re-run).

Derived deterministically from the committed transcript -- run it again any time the source
changes:
    python scripts/make_static_transcript.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_ID = "GDm_uH6VxPY"
SRC = REPO_ROOT / "data" / "demo" / VIDEO_ID / "transcript.json"
DST = REPO_ROOT / "static" / "demo" / VIDEO_ID / "transcript.json"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source transcript {SRC}")
    src = json.loads(SRC.read_text())
    segments = [
        {"start": round(float(s["start"]), 2), "text": s["text"].strip()}
        for s in src["segments"]
        if s.get("text", "").strip()
    ]
    DST.write_text(json.dumps({"videoId": VIDEO_ID, "segments": segments}) + "\n")
    size = DST.stat().st_size
    print(f"wrote {DST.relative_to(REPO_ROOT)}  ({len(segments)} segments, {size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
