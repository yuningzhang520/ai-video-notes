"""Step 4C: build the full demo Note with ONE structure pass.

    transcript + the committed VisualBlocks
        -> ONE Claude section-plan call  (section_plan.json / structure_raw.json)
        -> deterministic assembly         (note_full.json)

The model only plans (topics + which blocks/anchors go where); the assembly copies the real
VisualBlocks UNCHANGED and derives anchor timestamps from frame_analysis' verify-lead. This
is build-time + committed: the deployed app serves note_full.json with NO request-time call.

Cost guard: if section_plan.json exists it is REUSED (no Anthropic call) and note_full.json
is re-assembled deterministically; pass --force to re-call the model.

Run from the repo root (needs ANTHROPIC_API_KEY in env or .env):
    python scripts/build_full_note.py  [--force]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.grounding import bullet_tier  # noqa: E402
from app.models import Note  # noqa: E402
from app.structure import (  # noqa: E402
    DEFAULT_MODEL,
    assemble_note,
    block_summaries,
    request_section_plan,
)


def _preview(text: str, n: int = 80) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")

VIDEO_ID = "GDm_uH6VxPY"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
VIDEO_TITLE = "AI agent design patterns"  # the on-screen title slide (frame-0002)

DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANALYSIS_PATH = DATA_DIR / "frame_analysis.json"
BLOCKS_PATH = DATA_DIR / "visual_blocks.json"
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
PLAN_PATH = DATA_DIR / "section_plan.json"
RAW_PATH = DATA_DIR / "structure_raw.json"
NOTE_PATH = DATA_DIR / "note_full.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def get_plan(manifest: dict, transcript: dict, summaries: list[dict], force: bool) -> dict:
    """Return the section plan, calling Anthropic at most once (cached unless --force)."""
    if PLAN_PATH.exists() and not force:
        print(f"[structure] reusing cached plan {PLAN_PATH.relative_to(REPO_ROOT)} (no API call; --force to refresh)")
        return load_json(PLAN_PATH)

    print(f"[structure] ONE call: {len(summaries)} blocks + {transcript['segmentCount']} transcript segments")
    result = request_section_plan(VIDEO_TITLE, float(manifest["durationSec"]),
                                  transcript["segments"], summaries)
    RAW_PATH.write_text(json.dumps({
        "model": result.model,
        "inputTokens": result.input_tokens,
        "outputTokens": result.output_tokens,
        "ok": result.ok,
        "error": result.error,
        "raw": result.raw,  # model text only -- no keys/headers
    }, indent=2) + "\n")
    if not result.ok:
        raise SystemExit(f"[structure] FAILED: {result.error}\n(wrote {RAW_PATH.relative_to(REPO_ROOT)} for inspection)")
    PLAN_PATH.write_text(json.dumps(result.plan, indent=2) + "\n")
    print(f"[structure] ok: {len(result.plan['sections'])} sections; wrote {PLAN_PATH.relative_to(REPO_ROOT)}")
    return result.plan


def main() -> None:
    force = "--force" in sys.argv[1:]
    for path in (MANIFEST_PATH, ANALYSIS_PATH, BLOCKS_PATH, TRANSCRIPT_PATH):
        if not path.exists():
            raise SystemExit(f"missing artifact {path} -- run Steps 4A/4B first")

    manifest = load_json(MANIFEST_PATH)
    frame_analysis = load_json(ANALYSIS_PATH)
    visual_blocks = load_json(BLOCKS_PATH)
    transcript = load_json(TRANSCRIPT_PATH)

    summaries = block_summaries(visual_blocks)
    plan = get_plan(manifest, transcript, summaries, force)

    video_meta = {"id": VIDEO_ID, "title": VIDEO_TITLE, "url": VIDEO_URL,
                  "durationSec": manifest["durationSec"]}
    note, rep = assemble_note(plan, manifest, frame_analysis, visual_blocks,
                              transcript["segments"], video_meta, DEFAULT_MODEL)

    NOTE_PATH.write_text(json.dumps(note.model_dump(by_alias=True), indent=2) + "\n")
    # Re-validate the written file against the schema (belt and suspenders).
    Note.model_validate_json(NOTE_PATH.read_text())

    # --- Report ------------------------------------------------------------------
    ids = {f.id for f in note.frames}
    tiers = rep.bullets_by_tier
    print("\n=== note_full.json ===")
    print(f"  video:    {note.video.id}  \"{note.video.title}\"  ({note.video.duration_sec}s)")
    print(f"  frames:   {len(note.frames)} (registry, all /static/demo/...)")
    print(f"  sections: {rep.section_count}   "
          f"bullets: {sum(rep.bullets_per_section.values())} "
          f"(visual {tiers.get('visual', 0)} / spoken {tiers.get('spoken', 0)} / "
          f"ungrounded {tiers.get('ungrounded', 0)})")
    print(f"  blocks placed: {len(rep.placed_block_refs)}/{len(summaries)}"
          f"   deduped: {len(rep.deduped_block_refs)}\n")

    for sec in note.sections:
        print(f"  [{sec.id}] {sec.title}")
        for bi, b in enumerate(sec.bullets):
            tier = bullet_tier(b, ids)
            blk = ", ".join(f"{x.frame_ref}:{x.type}/{x.confidence}" for x in b.blocks) or "-"
            if tier == "visual":
                src = f"frame {b.anchor.frame_ref} @ {b.anchor.timestamp_sec:.0f}s"
            elif tier == "spoken":
                src = f"spoken @ {b.anchor.timestamp_sec:.0f}s"
            else:
                src = "no source"
            print(f"      {bi}. [{tier}] {_preview(b.text)}")
            print(f"         blocks: {blk}   verify: {src}")

    if rep.dedup_drops:
        print("\n  DEDUPED (same-bullet near-duplicate blocks, fuller one kept):")
        for d in rep.dedup_drops:
            print(f"    dropped {d['dropped']}  (kept {d['kept']}, line-overlap {d['overlap']})")

    if rep.unplaced_block_refs:
        print("\n  UNPLACED blocks (reported, not forced in):")
        for r in rep.unplaced_block_refs:
            print(f"    {r}: {rep.unplaced_reasons.get(r, '')}")
    else:
        print("\n  all blocks placed")

    if rep.issues:
        print("\n  assembly issues:")
        for it in rep.issues:
            print(f"    - {it}")

    print(f"\n  wrote: {NOTE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
