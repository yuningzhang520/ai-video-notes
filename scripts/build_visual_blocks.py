"""Step 4B: run build-time Claude vision over the Step 4A vision candidates.

Reads the committed Step 4A artifact (frame_analysis.json) and runs ONE Claude vision call
per DETERMINISTIC candidate -- only frames where ``isVisionCandidate`` and ``isRepresentative``
are both true, never the non-candidates. Each frame yields AT MOST ONE
VisualBlock (the single most salient artifact), and the extracted blocks are committed to:

    data/demo/GDm_uH6VxPY/visual_blocks.json

Same build-time-then-commit posture as Step 3B: the deployed app reads this artifact; it
does NOT call Anthropic at request time. This step does NOT build the final Note (that is
Step 4C's structure pass).

Reuse / cost guard:
  - Uses the SAME single client as Step 3B (``app.vision.extract_block``); no second client.
  - Model is the ``ANTHROPIC_MODEL`` env override, default ``claude-sonnet-4-6``.
  - If visual_blocks.json already has a SUCCESSFUL block for a frame, it is reused with NO
    API call (resume). A plain re-run on a complete artifact makes zero calls.
  - ``--force`` re-calls every candidate from scratch.

Hints (CONTEXT ONLY -- the image is ground truth): each call is given that frame's 4A
``ocrText`` and ``candidateReason`` plus nearby transcript. The model is told to correct/
complete the OCR, never copy it verbatim (OCR is noisy/truncated).

Run from the repo root (needs ANTHROPIC_API_KEY in env or .env):
    python scripts/build_visual_blocks.py  [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.frame_select import phash_distance  # noqa: E402
from app.vision import (  # noqa: E402
    DEFAULT_MODEL,
    VALID_CONFIDENCE,
    VALID_TYPES,
    VisionResult,
    extract_block,
)

VIDEO_ID = "GDm_uH6VxPY"
DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANALYSIS_PATH = DATA_DIR / "frame_analysis.json"
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
BLOCKS_PATH = DATA_DIR / "visual_blocks.json"

# Vision-reuse reference: a snapshot of the PREVIOUS frame_analysis + visual_blocks (taken before
# this rerun overwrote them). Present only during a re-sample; absent on a first build.
REUSE_REF_DIR = REPO_ROOT / "work" / "reuse_ref"

# --- Vision-reuse guard (Step 6 rerun) ----------------------------------------------------
# When the frame set is re-sampled, a candidate that is near-identical to a frame we already
# paid to extract should REUSE that extraction instead of re-calling the model. The match is
# deliberately TIGHT (false-reuse -- attaching the wrong content -- is far worse than false-miss,
# which just re-pays a few cents): full-frame pHash <= 2 AND |dt| <= 8s. The pHash compared is the
# OLD frame_analysis pHash (computed at 640x360), NEVER the on-disk frame -- diagram frames are
# 1080p on disk, so re-hashing them would never match a new 640x360 candidate.
REUSE_MAX_HAMMING = 2
REUSE_MAX_DT = 8.0

TRANSCRIPT_WINDOW_SEC = 20.0   # nearby-transcript context window, +/- around the frame
CALL_SPACING_SEC = 0.5         # gentle spacing between calls (a small batch)

# Rough public list pricing for the Sonnet tier, USD per 1M tokens -- for an APPROXIMATE
# cost estimate only (the real number is on your Anthropic dashboard).
COST_PER_MTOK_IN = 3.0
COST_PER_MTOK_OUT = 15.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def is_success(entry: dict) -> bool:
    return entry.get("error") is None and bool((entry.get("content") or "").strip())


def load_reuse_reference() -> tuple[list[dict], dict[str, str]]:
    """Return (old successful blocks, old pHash-by-frameRef) from the snapshot, or ([], {}) if no
    reference is present. The pHash is the OLD frame_analysis pHash (640x360) -- never the on-disk
    frame -- so the diagram frame (1080p on disk) still matches a fresh 640x360 candidate."""
    fa, vb = REUSE_REF_DIR / "frame_analysis.json", REUSE_REF_DIR / "visual_blocks.json"
    if not (fa.is_file() and vb.is_file()):
        return [], {}
    old_phash = {f["id"]: f["phash"] for f in load_json(fa)["frames"]}
    old_blocks = [b for b in load_json(vb).get("blocks", []) if is_success(b)]
    return old_blocks, old_phash


def match_reusable_block(candidate: dict, old_blocks: list[dict],
                         old_phash_by_ref: dict[str, str],
                         max_hamming: int = REUSE_MAX_HAMMING,
                         max_dt: float = REUSE_MAX_DT) -> dict | None:
    """Return the old SUCCESSFUL block whose source frame is near-identical to ``candidate``
    (full-frame pHash <= max_hamming AND |dt| <= max_dt), else None. ``candidate`` needs
    ``phash`` + ``timestampSec``. Ties break to the smallest (pHash, |dt|)."""
    best = None
    for b in old_blocks:
        oph = old_phash_by_ref.get(b["frameRef"])
        if oph is None:
            continue
        d = phash_distance(candidate["phash"], oph)
        dt = abs(float(b["timestampSec"]) - float(candidate["timestampSec"]))
        if d <= max_hamming and dt <= max_dt and (best is None or (d, dt) < best[0]):
            best = ((d, dt), b)
    return best[1] if best else None


def remap_reused_block(old_block: dict, candidate: dict) -> dict:
    """Reuse the old block's VISION content (type/content/confidence) but repoint EVERY
    frame-derived field at the NEW representative -- no stale old frameRef survives anywhere."""
    return {
        "frameRef": candidate["id"],
        "timestampSec": candidate["timestampSec"],
        "suggestedVerifyTimestampSec": candidate["suggestedVerifyTimestampSec"],
        "type": old_block["type"],
        "content": old_block["content"],
        "confidence": old_block["confidence"],
        "sourceImagePath": candidate["imagePath"],
        "ocrText": candidate.get("ocrText", ""),
        "model": old_block.get("model"),
        "inputTokens": 0,
        "outputTokens": 0,
        "error": None,
        "reusedFromFrameRef": old_block["frameRef"],
    }


def nearby_transcript(segments: list[dict], ts: float, window: float = TRANSCRIPT_WINDOW_SEC) -> str:
    """Join transcript text within +/- `window` seconds of `ts` (context for the call)."""
    return " ".join(
        s["text"].strip()
        for s in segments
        if (ts - window) <= s["start"] <= (ts + window)
    ).strip()


def select_candidates(analysis: dict) -> list[dict]:
    """The Step 4A vision candidates ONLY: representatives that cleared the text gate."""
    return [
        f for f in analysis["frames"]
        if f.get("isVisionCandidate") and f.get("isRepresentative")
    ]


def classify_result(candidate: dict, result: VisionResult) -> dict:
    """Turn a vision result into ONE visual-block entry, strictly classifying success/failure.

    A successful block requires ``ok`` AND a valid type, a valid confidence, and non-empty
    content. An empty content string, an invalid type, an invalid confidence, or an upstream
    error (missing key/file, API error, unparsable JSON) all produce a FAILED entry carrying
    an ``error`` -- never a successful block. Always exactly one entry per frame.
    """
    entry = {
        "frameRef": candidate["id"],
        "timestampSec": candidate["timestampSec"],
        "suggestedVerifyTimestampSec": candidate["suggestedVerifyTimestampSec"],
        "type": result.type,
        "content": result.content,
        "confidence": result.confidence,
        "sourceImagePath": candidate["imagePath"],
        "ocrText": candidate["ocrText"],
        "model": result.model,
        "inputTokens": result.input_tokens,
        "outputTokens": result.output_tokens,
        "error": None,
    }

    if not result.ok:
        entry["error"] = result.error or "vision call failed"
        return entry

    problems = []
    if result.type not in VALID_TYPES:
        problems.append(f"invalid type {result.type!r}")
    if result.confidence not in VALID_CONFIDENCE:
        problems.append(f"invalid confidence {result.confidence!r}")
    if not (result.content and result.content.strip()):
        problems.append("empty content")
    if problems:
        entry["error"] = "invalid block: " + ", ".join(problems)
    return entry


def estimate_cost_usd(blocks: list[dict]) -> float:
    in_tok = sum(b.get("inputTokens") or 0 for b in blocks)
    out_tok = sum(b.get("outputTokens") or 0 for b in blocks)
    return in_tok / 1e6 * COST_PER_MTOK_IN + out_tok / 1e6 * COST_PER_MTOK_OUT


def preview(text: str | None, n: int = 70) -> str:
    one_line = " ".join((text or "").split())
    return one_line[:n] + ("…" if len(one_line) > n else "")


def main() -> None:
    force = "--force" in sys.argv[1:]
    for path in (MANIFEST_PATH, ANALYSIS_PATH, TRANSCRIPT_PATH):
        if not path.exists():
            raise SystemExit(f"missing artifact {path} -- run Steps 3B-prep / 4A first")

    manifest = load_json(MANIFEST_PATH)
    analysis = load_json(ANALYSIS_PATH)
    transcript = load_json(TRANSCRIPT_PATH)
    segments = transcript["segments"]
    frame_ids = {f["id"] for f in manifest["frames"]}

    candidates = select_candidates(analysis)
    if not candidates:
        raise SystemExit("no vision candidates in frame_analysis.json -- rerun Step 4A")
    # Every candidate must be a real frame; we only ever process these.
    bad = [c["id"] for c in candidates if c["id"] not in frame_ids]
    if bad:
        raise SystemExit(f"candidate(s) not in manifest: {bad}")

    cached: dict[str, dict] = {}
    if BLOCKS_PATH.exists() and not force:
        for b in load_json(BLOCKS_PATH).get("blocks", []):
            if is_success(b):
                cached[b["frameRef"]] = b

    # Vision-reuse guard: a candidate near-identical to a frame we already paid to extract reuses
    # that extraction (content remapped to the new frame) instead of re-calling the model.
    old_blocks, old_phash = ([], {}) if force else load_reuse_reference()

    print(f"=== Step 4B vision over {len(candidates)} candidate(s) ===")
    print(f"  model:   {DEFAULT_MODEL} (override via ANTHROPIC_MODEL)")
    print(f"  cache:   {len(cached)} reusable success block(s)"
          f"{' (ignored: --force)' if force else ''}")
    print(f"  reuse:   {len(old_blocks)} prior block(s) available "
          f"(pHash <= {REUSE_MAX_HAMMING}, |dt| <= {REUSE_MAX_DT:.0f}s)\n")

    blocks: list[dict] = []
    calls = 0
    reused = 0
    total_latency = 0.0
    for cand in candidates:
        fref = cand["id"]
        # Resume-cache, VALIDATED by content: a re-sample reuses frame ids for DIFFERENT frames,
        # so trust a cached block only if its ocrText still matches this candidate's (same frame).
        prior = cached.get(fref)
        if prior is not None and not force and (prior.get("ocrText") or "") == (cand.get("ocrText") or ""):
            blocks.append(prior)
            print(f"  {fref} @ {cand['timestampSec']:>5.0f}s  CACHED  "
                  f"type={prior['type']} conf={prior['confidence']}")
            continue

        match = match_reusable_block(cand, old_blocks, old_phash) if old_blocks else None
        if match is not None:
            entry = remap_reused_block(match, cand)
            blocks.append(entry)
            reused += 1
            print(f"  {fref} @ {cand['timestampSec']:>5.0f}s  REUSED <- {match['frameRef']}  "
                  f"type={entry['type']} conf={entry['confidence']} (no API call)")
            continue

        if calls and CALL_SPACING_SEC:
            time.sleep(CALL_SPACING_SEC)
        disk_path = REPO_ROOT / cand["imagePath"].lstrip("/")
        ctx = nearby_transcript(segments, float(cand["timestampSec"]))
        t0 = time.perf_counter()
        result = extract_block(
            disk_path,
            transcript_context=ctx,
            ocr_hint=cand.get("ocrText", ""),
            candidate_reason=cand.get("candidateReason", ""),
        )
        dt = time.perf_counter() - t0
        calls += 1
        total_latency += dt

        entry = classify_result(cand, result)
        entry["latencySec"] = round(dt, 2)
        blocks.append(entry)

        status = "ok    " if is_success(entry) else "FAILED"
        tail = preview(entry["content"]) if is_success(entry) else (entry["error"] or "")
        print(f"  {fref} @ {cand['timestampSec']:>5.0f}s  {status}  "
              f"type={entry['type']} conf={entry['confidence']} {dt:>4.1f}s | {tail}")

    blocks.sort(key=lambda b: float(b["timestampSec"]))
    successes = [b for b in blocks if is_success(b)]
    failures = [b for b in blocks if not is_success(b)]
    # Composition is derived from the BLOCKS (not the run): a reused block carries reusedFromFrameRef.
    reused_blocks = [b for b in successes if b.get("reusedFromFrameRef")]
    fresh_blocks = [b for b in successes if not b.get("reusedFromFrameRef")]

    out = {
        "videoId": VIDEO_ID,
        "sources": {
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "frameAnalysis": str(ANALYSIS_PATH.relative_to(REPO_ROOT)),
            "transcript": str(TRANSCRIPT_PATH.relative_to(REPO_ROOT)),
        },
        "model": DEFAULT_MODEL,
        "params": {
            "transcriptWindowSec": TRANSCRIPT_WINDOW_SEC,
            "atMostOneBlockPerFrame": True,
        },
        "counts": {
            "candidates": len(candidates),
            "successfulBlocks": len(successes),
            "failedBlocks": len(failures),
            "reusedBlocks": len(reused_blocks),
            "freshBlocks": len(fresh_blocks),
            "freshVisionCallsThisRun": calls,
        },
        "blocks": blocks,
    }
    BLOCKS_PATH.write_text(json.dumps(out, indent=2) + "\n")

    est = estimate_cost_usd([b for b in blocks if b.get("inputTokens")])
    print("\n=== summary ===")
    print(f"  candidates:        {len(candidates)}")
    print(f"  successful blocks: {len(successes)}")
    print(f"  failed blocks:     {len(failures)}"
          + (f"  ({[b['frameRef'] for b in failures]})" if failures else ""))
    print(f"  reused (no API):   {len(reused_blocks)}  (remapped from prior blocks; reusedFromFrameRef)")
    print(f"  fresh-visioned:    {len(fresh_blocks)}   (API calls this run: {calls})")
    if calls:
        print(f"  latency:           {total_latency:.1f}s total, {total_latency / calls:.1f}s/call avg")
    print(f"  est. cost:         ~${est:.4f} (approx, Sonnet list pricing on captured usage)")
    print(f"  wrote:             {BLOCKS_PATH.relative_to(REPO_ROOT)}")
    by_type: dict[str, int] = {}
    for b in successes:
        by_type[b["type"]] = by_type.get(b["type"], 0) + 1
    print(f"  block types:       {by_type}")


if __name__ == "__main__":
    main()
