"""Step 4C/5A structure pass: transcript + extracted blocks -> a topic-segmented Note.

The substance is a list of BULLETS per section (Step 5A) -- each a self-contained point that
is individually verifiable. Two halves, deliberately separated so the model NEVER touches
verifiable content:

  1. ``request_section_plan`` -- ONE Claude Messages call. It sees the transcript and a
     COMPACT one-line summary of each VisualBlock (frameRef / type / time / ~110-char gist)
     and returns ONLY a section PLAN: logical topics with bullets, where each bullet says how
     it is grounded -- by existing block(s) (visual), by a cited transcript second (spoken),
     or neither (ungrounded). It does NOT see or reproduce block content, and it owns the
     SEMANTIC FILTER (junk blocks -> unplacedBlockRefs).

  2. ``assemble_note`` -- pure, deterministic, no network. It copies each referenced
     VisualBlock UNCHANGED from visual_blocks.json into its bullet, builds one frames registry
     from the manifest, and derives every anchor timestamp deterministically -- a visual
     bullet from frame_analysis' ``suggestedVerifyTimestampSec`` (the verify-lead), a spoken
     bullet by snapping the cited second to the nearest real transcript segment.start. A block
     attaches to at most one bullet (first wins); unknown refs are dropped with an issue; a
     bullet with no resolvable source stays ungrounded (anchor=None) without dropping its text.

Same client/model pattern as app/vision.py (ANTHROPIC_MODEL default claude-sonnet-4-6;
load_dotenv is a LOCAL convenience only). Build-time: scripts/build_full_note.py runs this
once and commits note_full.json; the deployed app reads it and never calls Anthropic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .grounding import check_note_grounding
from .models import Anchor, Bullet, Frame, Note, Section, Video, VisualBlock
from .textsim import (
    line_overlap,
    more_complete_text,
    norm_lines,
    symmetric_ratio,
    text_prefix_subset,
    token_jaccard,
)

try:  # pragma: no cover - best-effort local .env load, exactly as app/vision.py does
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
GIST_CHARS = 110           # how much of a block's content the planner sees (a hint, not the content)
MIN_BULLETS = 1            # per section: 1 only if the topic is genuinely a single point
MAX_BULLETS = 7            # per section: cap to avoid over-splitting
TRANSCRIPT_SPAN_SEC = 8.0  # spoken-bullet caption span: snapped segment + this many seconds
SPAN_CHAR_CAP = 200        # ... capped to a readable quote
DEDUP_TEXT_OVERLAP = 0.75  # within a bullet, same-type blocks at/above this line-overlap are
                           # near-duplicates. Deliberately conservative (a false merge that
                           # loses unique content is the worse error -- same reasoning as 4A's
                           # pHash threshold); the real line-by-line wrap-up case overlaps ~1.0.

# Section-scoped cross-bullet dedup (Step 6 follow-up): fold a block that is a clear near-SUBSET of
# a fuller SAME-type block elsewhere in the SAME section -- e.g. one scroll-view of a code file
# when the full file is shown in another bullet. STRICTER than the within-bullet pass and
# type-specific, because folding across bullets is more aggressive. Conservative bias throughout:
# when in doubt, keep both; never fold two blocks that each carry distinct content.
SECTION_DEDUP_TEXT_OVERLAP = 0.85          # text: clear near-superset
SECTION_DEDUP_CODE_TABLE_OVERLAP = 0.90    # code/table: only a VERY clear containment (or true subset)
SECTION_DEDUP_DIAGRAM_OVERLAP = 0.90       # diagram: near-identical extracted text AND...
SECTION_DEDUP_DIAGRAM_IMG_HAMMING = 6      # ...near-identical images (pHash) -- else keep both
SECTION_DEDUP_CODE_RATIO = 0.75            # code ONLY: difflib similarity (symmetrized -- see
                                           # symmetric_ratio) on indentation-normalized text -- folds
                                           # a same-screen re-capture to the line-fuller block,
                                           # accepting the dropped block's few unique lines.
SECTION_DEDUP_JACCARD = 0.65               # text + diagram: token-set Jaccard on the extracted
                                           # content. Order-independent and immune to framing wording
                                           # (which pollutes char-difflib). Folds a same-content
                                           # re-capture to the line-fuller block. The Jin Sho text
                                           # recaptures land 0.66-0.76; the two trace diagrams, which
                                           # genuinely differ (collapsed vs expanded tree), stay at
                                           # 0.596 -- below the bar, so they do NOT fold.

_PROMPT = """You are turning ONE technical talk into topic-segmented, individually-verifiable \
notes. You are given the transcript and a list of already-extracted on-screen VISUAL BLOCKS \
(code/table/diagram/text), each read off a specific frame and identified by its frameRef.

Video: "{title}" ({duration} long).

TRANSCRIPT (each line is "[m:ss | <seconds>s] text"):
{transcript}

EXTRACTED VISUAL BLOCKS (frameRef | type | time | short gist -- the gist is a HINT ONLY):
{blocks}

Produce a SECTION PLAN as strict JSON. Each section is a LOGICAL TOPIC (not a time-chunk), \
ordered as the talk flows, and its substance is a list of BULLETS.

BULLETS -- the primary content:
- Each bullet is ONE self-contained, complete logical point, stated as a full proposition \
(the claim AND its mechanism / example / why) -- NOT a headline fragment, NOT a paragraph.
- Produce the points DIRECTLY. Capture EVERYTHING a thorough prose summary of the topic would \
contain -- lose no substance -- but as discrete points, not prose.
- {min_b}-{max_b} bullets per section, broken down by whatever structure fits THAT topic (no \
fixed template). Use 1 only if the topic genuinely has a single point; do not over-split.

Each bullet is grounded in EXACTLY ONE of three ways, chosen by what you put on it:
- VISUAL: put one or more "blockRefs" (existing blocks that support THIS exact point). The \
bullet is verified by its frame(s).
- SPOKEN: put a "transcriptCiteSec" -- an integer second copied from a transcript line above, \
near where this point is said. The bullet is verified by that spoken moment. Do NOT invent a \
number; copy a <seconds> value shown in the transcript.
- UNGROUNDED: neither -- ONLY if the point genuinely has no on-screen or spoken source (rare).

Block rules:
- Each blockRef attaches to AT MOST ONE bullet -- its single most-relevant one (use the \
block's time + content to choose). NEVER repeat a block across bullets.
- A bullet may carry 0, 1, or MULTIPLE blocks (e.g. a code block AND a diagram for one point).
- A valuable block that no spoken point really covers becomes its OWN visual bullet, with a \
short factual text describing what it shows.
- SEMANTIC FILTER: if a block has no meaningful teaching content -- e.g. a raw JSON / debug / \
trace dump that only passed an upstream character-count gate -- do NOT attach it and do NOT \
make it a bullet. List it in "unplacedBlockRefs" with a reason starting "junk: ".

Hard rules:
- Do NOT reproduce, transcribe, or paraphrase block CONTENT into bullet text. The gist is a \
hint; the bullet is YOUR synthesized point; the block travels separately as evidence.
- Use ONLY frameRefs from the block list and <seconds> from the transcript above. Invent neither.
- "gist" is a 1-sentence spoken summary of the section. Do NOT output block content or a full Note.

Respond with ONLY this JSON object, no prose and no markdown fences (per bullet include \
"blockRefs" OR "transcriptCiteSec" OR neither -- omit the keys that do not apply):
{{"sections": [{{"id": "kebab-id", "title": "...", "gist": "...", "bullets": [{{"text": "...", \
"blockRefs": ["..."], "transcriptCiteSec": 123}}]}}], "unplacedBlockRefs": ["..."], \
"unplacedReasons": {{"frameRef": "junk: ..."}}}}"""


@dataclass
class StructureResult:
    ok: bool = False
    plan: dict | None = None
    model: str | None = None
    error: str | None = None
    raw: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class AssemblyReport:
    section_count: int = 0
    bullets_per_section: dict[str, int] = field(default_factory=dict)   # section id -> bullet count
    bullets_by_tier: dict[str, int] = field(default_factory=dict)       # visual / spoken / ungrounded
    placed_block_refs: list[str] = field(default_factory=list)          # blocks actually in the note
    deduped_block_refs: list[str] = field(default_factory=list)         # attached then dropped as near-dups
    dedup_drops: list[dict] = field(default_factory=list)               # [{dropped, kept, overlap}]
    unplaced_block_refs: list[str] = field(default_factory=list)        # never attached by the planner
    unplaced_reasons: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def _mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _one_line(text: str, n: int = GIST_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")


def block_summaries(visual_blocks: dict) -> list[dict]:
    """Compact, content-free hints for the planner: one line per SUCCESSFUL block."""
    out = []
    for b in visual_blocks.get("blocks", []):
        if b.get("error") is not None or not (b.get("content") or "").strip():
            continue
        out.append({
            "frameRef": b["frameRef"],
            "type": b["type"],
            "timestampSec": b["timestampSec"],
            "gist": _one_line(b["content"]),
        })
    return out


def _parse_json(text: str) -> dict:
    """Strip a wrapping code fence if present, then parse; fall back to the first {...}."""
    t = (text or "").strip()
    fenced = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```$", t, re.DOTALL)
    if fenced:
        t = fenced.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        return json.loads(t[i : j + 1])
    raise ValueError("no JSON object found in model output")


def _build_prompt(title: str, duration_sec: float, segments: list[dict],
                  summaries: list[dict], min_bullets: int, max_bullets: int) -> str:
    transcript = "\n".join(
        f"[{_mmss(s['start'])} | {int(round(s['start']))}s] {s['text'].strip()}" for s in segments
    )
    blocks = "\n".join(
        f"- {b['frameRef']} | {b['type']} | {_mmss(b['timestampSec'])} | {b['gist']}"
        for b in summaries
    )
    return _PROMPT.format(
        title=title, duration=_mmss(duration_sec), transcript=transcript, blocks=blocks,
        min_b=min_bullets, max_b=max_bullets,
    )


def request_section_plan(title: str, duration_sec: float, segments: list[dict],
                         summaries: list[dict], min_bullets: int = MIN_BULLETS,
                         max_bullets: int = MAX_BULLETS,
                         model: str | None = None) -> StructureResult:
    """ONE Claude call -> a parsed section plan. NEVER raises; ok=False carries the cause."""
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    res = StructureResult(model=model)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        res.error = "ANTHROPIC_API_KEY not set (looked in environment and .env)"
        return res
    try:
        import anthropic
    except ImportError as e:
        res.error = f"anthropic SDK not installed ({e})"
        return res

    prompt = _build_prompt(title, duration_sec, segments, summaries, min_bullets, max_bullets)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
    except Exception as e:
        res.error = f"{type(e).__name__}: {getattr(e, 'message', e)}"
        return res

    usage = getattr(message, "usage", None)
    res.input_tokens = getattr(usage, "input_tokens", None)
    res.output_tokens = getattr(usage, "output_tokens", None)
    res.raw = "".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()

    try:
        plan = _parse_json(res.raw)
    except Exception as e:
        res.error = f"could not parse plan as JSON: {e}"
        return res
    if not isinstance(plan.get("sections"), list) or not plan["sections"]:
        res.error = "plan has no sections"
        return res
    res.plan, res.ok = plan, True
    return res


def _snap_segment(cite_sec: float, segments: list[dict]) -> dict:
    """The transcript segment nearest a cited second; ties break to the earlier segment."""
    return min(segments, key=lambda s: (abs(float(s["start"]) - cite_sec), float(s["start"])))


def _span_text(start: float, segments: list[dict],
               span_sec: float = TRANSCRIPT_SPAN_SEC, cap: int = SPAN_CHAR_CAP) -> str:
    """A readable caption quote: the snapped segment + following segments within span_sec."""
    chunk = [s["text"].strip() for s in segments
             if start <= float(s["start"]) < start + span_sec and s["text"].strip()]
    text = " ".join(chunk).strip()
    return text[:cap].rstrip() + ("…" if len(text) > cap else "")


def _more_complete(a: VisualBlock, b: VisualBlock) -> bool:
    """True iff `a` is strictly more complete than `b` (more lines, then longer text).

    Thin wrapper over ``textsim.more_complete_text`` on the blocks' content, so 5A's in-bullet
    dedup and Step 6's frame selection share one normalization/overlap rule.
    """
    return more_complete_text(a.content, b.content)


def dedup_blocks_in_bullet(blocks: list[VisualBlock]) -> tuple[list[VisualBlock], list[dict]]:
    """Drop near-duplicate blocks WITHIN one bullet, keeping the more complete one.

    Deterministic and content-based -- it catches what 4A's pHash cannot: a slide that reveals
    line-by-line produces frames that differ above the pHash threshold but whose extracted text
    is a near-superset. Only compares SAME-type blocks; only drops at >= DEDUP_TEXT_OVERLAP line
    overlap (conservative -- when in doubt it keeps both). It NEVER rewrites a kept block's
    content (blocks stay verbatim); it only drops a duplicate and reports it. Pure: no I/O.

    Returns ``(kept_blocks, drops)`` where each drop is ``{dropped, kept, overlap}`` (frameRefs).
    """
    kept: list[VisualBlock] = []
    drops: list[dict] = []
    for blk in blocks:
        dup_idx, overlap = None, 0.0
        for i, k in enumerate(kept):
            if k.type != blk.type:
                continue
            o = line_overlap(k.content, blk.content)
            if o >= DEDUP_TEXT_OVERLAP:
                dup_idx, overlap = i, o
                break
        if dup_idx is None:
            kept.append(blk)
            continue
        k = kept[dup_idx]
        if _more_complete(blk, k):  # the newcomer is fuller -> it replaces the kept block
            drops.append({"dropped": k.frame_ref, "kept": blk.frame_ref, "overlap": round(overlap, 3)})
            kept[dup_idx] = blk
        else:                       # the kept block is at least as complete -> drop the newcomer
            drops.append({"dropped": blk.frame_ref, "kept": k.frame_ref, "overlap": round(overlap, 3)})
    return kept, drops


def _phash_distance(a: str, b: str) -> int:
    """Hamming distance between two phash hex strings (popcount of the XOR). Inlined here to avoid
    importing app.frame_select (which pulls Tesseract/imagehash) into the runtime structure path."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _line_subset(small: str, large: str) -> bool:
    """True iff every normalized line of `small` appears in `large` (true containment)."""
    ls, ll = set(norm_lines(small)), set(norm_lines(large))
    return bool(ls) and ls <= ll


def _section_near_duplicate(a: VisualBlock, b: VisualBlock,
                            phash_by_id: dict[str, str]) -> tuple[bool, float]:
    """Type-specific near-subset test for the SECTION pass. Returns (is_duplicate, line_overlap).

    text     -> line-overlap >= SECTION_DEDUP_TEXT_OVERLAP, a typewriter/reveal PARTIAL (one is a
                literal flattened prefix of the other), a TRUE line-subset (0 unique lines), OR a
                token_jaccard >= SECTION_DEDUP_JACCARD (a same-content re-capture)
    code     -> line-overlap >= SECTION_DEDUP_CODE_TABLE_OVERLAP, a TRUE line-subset, OR an
                order-independent symmetric_ratio >= SECTION_DEDUP_CODE_RATIO on normalized text
    table    -> line-overlap >= SECTION_DEDUP_CODE_TABLE_OVERLAP, OR a TRUE line-subset
    diagram  -> token_jaccard >= SECTION_DEDUP_JACCARD on the extracted text, OR (line-overlap >=
                SECTION_DEDUP_DIAGRAM_OVERLAP AND the two FRAMES are near-identical images by pHash)
    Different types never match. The returned signal is the strongest one that fired.
    """
    if a.type != b.type:
        return False, 0.0
    o = line_overlap(a.content, b.content)
    subset = _line_subset(a.content, b.content) or _line_subset(b.content, a.content)
    if a.type == "text":
        partial = (subset
                   or text_prefix_subset(a.content, b.content)
                   or text_prefix_subset(b.content, a.content))
        if partial:
            return True, 1.0   # prefix/subset = full containment of the smaller (zero-loss fold)
        j = token_jaccard(a.content, b.content)
        return (o >= SECTION_DEDUP_TEXT_OVERLAP or j >= SECTION_DEDUP_JACCARD), max(o, j)
    if a.type == "code":
        r = symmetric_ratio(a.content, b.content)   # order-independent (min of both directions)
        return (subset or o >= SECTION_DEDUP_CODE_TABLE_OVERLAP or r >= SECTION_DEDUP_CODE_RATIO), max(o, r)
    if a.type == "table":
        return (subset or o >= SECTION_DEDUP_CODE_TABLE_OVERLAP), o
    if a.type == "diagram":
        j = token_jaccard(a.content, b.content)
        if j >= SECTION_DEDUP_JACCARD:
            return True, max(o, j)
        if o < SECTION_DEDUP_DIAGRAM_OVERLAP:
            return False, max(o, j)
        pa, pb = phash_by_id.get(a.frame_ref), phash_by_id.get(b.frame_ref)
        return (bool(pa) and bool(pb)
                and _phash_distance(pa, pb) <= SECTION_DEDUP_DIAGRAM_IMG_HAMMING), max(o, j)
    return False, o


def dedup_blocks_across_section(items: list[tuple[int, VisualBlock]],
                                phash_by_id: dict[str, str]) -> tuple[set[str], list[dict]]:
    """Drop a block that is a clear near-subset of a fuller SAME-type block elsewhere in the SAME
    section. ``items`` is ``[(bullet_index, block)]`` in reading order; the kept block may sit in
    any bullet. Returns ``(dropped_frame_refs, drops)`` where drops are ``{dropped, kept, overlap}``.
    NEVER compares across types; NEVER folds two blocks below the type bar (each keeps distinct
    content). Pure -- blocks stay verbatim; this only chooses which to drop.
    """
    kept: list[VisualBlock] = []
    drops: list[dict] = []
    dropped: set[str] = set()
    for _, blk in items:
        dup_idx, overlap = None, 0.0
        for i, k in enumerate(kept):
            is_dup, o = _section_near_duplicate(k, blk, phash_by_id)
            if is_dup:
                dup_idx, overlap = i, o
                break
        if dup_idx is None:
            kept.append(blk)
            continue
        k = kept[dup_idx]
        if _more_complete(blk, k):   # the newcomer is fuller -> keep it, drop the earlier one
            drops.append({"dropped": k.frame_ref, "kept": blk.frame_ref, "overlap": round(overlap, 3)})
            dropped.add(k.frame_ref)
            kept[dup_idx] = blk
        else:                        # the kept block is at least as complete -> drop the newcomer
            drops.append({"dropped": blk.frame_ref, "kept": k.frame_ref, "overlap": round(overlap, 3)})
            dropped.add(blk.frame_ref)
    return dropped, drops


def assemble_note(plan: dict, manifest: dict, frame_analysis: dict, visual_blocks: dict,
                  transcript_segments: list[dict], video_meta: dict,
                  model: str) -> tuple[Note, AssemblyReport]:
    """Deterministically build the Note from the plan + the UNCHANGED committed artifacts.

    For each planned bullet:
      - copy any referenced blocks VERBATIM from visual_blocks.json (type/content/confidence/
        frameRef); a block attaches to AT MOST ONE bullet across the whole note (first wins);
      - choose the anchor deterministically: VISUAL if the bullet has block(s) (kind="frame",
        ts = the primary block's suggestedVerifyTimestampSec verify-lead); else SPOKEN if it
        cites a transcript second (kind="transcript", snapped to the nearest real
        segment.start, carrying that caption span); else UNGROUNDED (anchor=None, text kept --
        never silently trusted).
    Unknown / already-used block refs are dropped with an issue. The model owns the semantic
    filter (junk blocks land in plan.unplacedBlockRefs); assembly just reports them.
    """
    rep = AssemblyReport()

    blocks_by_ref = {b["frameRef"]: b for b in visual_blocks.get("blocks", [])
                     if b.get("error") is None and (b.get("content") or "").strip()}
    all_block_refs = list(blocks_by_ref)
    # All frames carry a suggestedVerifyTimestampSec in frame_analysis (max(ts-3, 0)).
    suggested = {f["id"]: f["suggestedVerifyTimestampSec"] for f in frame_analysis["frames"]}
    ocr_by_id = {f["id"]: f.get("ocrText") for f in frame_analysis["frames"]}
    phash_by_id = {f["id"]: f.get("phash") for f in frame_analysis["frames"]}  # for diagram-image check
    manifest_frames = manifest["frames"]

    used_refs: set[str] = set()   # every block claimed by a bullet (one-block-<=1-bullet guard)
    kept_refs: set[str] = set()   # blocks that survive same-bullet dedup -> actually in the note
    sections: list[Section] = []

    for i, sp in enumerate(plan.get("sections", [])):
        sid = (sp.get("id") or f"sec-{i + 1}").strip() or f"sec-{i + 1}"

        # Phase 1 -- resolve each bullet's blocks (verbatim) + the within-bullet dedup. Anchors are
        # deferred to phase 3 because the section-scoped pass (phase 2) may still drop a block.
        pending: list[dict] = []
        for bi, bp in enumerate(sp.get("bullets", []) or []):
            text = (bp.get("text") or "").strip()
            if not text:
                rep.issues.append(f"section {sid} bullet {bi}: empty text; dropped")
                continue

            attached: list[VisualBlock] = []
            for ref in bp.get("blockRefs", []) or []:
                if ref not in blocks_by_ref:
                    rep.issues.append(f"section {sid} bullet {bi}: unknown blockRef {ref!r} dropped")
                    continue
                if ref in used_refs:
                    rep.issues.append(f"section {sid} bullet {bi}: blockRef {ref!r} already placed; dropped")
                    continue
                used_refs.add(ref)
                b = blocks_by_ref[ref]
                attached.append(VisualBlock(
                    type=b["type"], content=b["content"],
                    frame_ref=b["frameRef"], confidence=b.get("confidence"),
                ))

            blk_objs, drops = dedup_blocks_in_bullet(attached)
            for d in drops:
                rep.dedup_drops.append(d)
                rep.issues.append(
                    f"section {sid} bullet {bi}: dropped near-duplicate block {d['dropped']} "
                    f"(overlap {d['overlap']}; kept fuller block {d['kept']})"
                )
            pending.append({"text": text, "blocks": blk_objs,
                            "origRefs": [b.frame_ref for b in blk_objs],
                            "cite": bp.get("transcriptCiteSec")})

        # Phase 2 -- section-scoped cross-bullet dedup: drop a clear near-subset of a fuller
        # same-type block elsewhere in THIS section (e.g. a code scroll-view when the full file is
        # shown in another bullet). Conservative + type-specific; blocks stay verbatim.
        flat = [(idx, b) for idx, pb in enumerate(pending) for b in pb["blocks"]]
        sec_dropped, sec_drops = dedup_blocks_across_section(flat, phash_by_id)
        fold_target = {d["dropped"]: d["kept"] for d in sec_drops}   # dropped ref -> the fuller ref kept
        for d in sec_drops:
            rep.dedup_drops.append(d)
            rep.issues.append(
                f"section {sid}: dropped cross-bullet near-duplicate block {d['dropped']} "
                f"(overlap {d['overlap']}; kept fuller block {d['kept']} elsewhere in section)"
            )
        for pb in pending:
            pb["blocks"] = [b for b in pb["blocks"] if b.frame_ref not in sec_dropped]

        # Phase 3 -- finalize: anchor each bullet from its SURVIVING blocks (visual > spoken >
        # ungrounded). A bullet whose ONLY block was section-folded into a fuller block stays
        # clickable -- it verifies against that fuller block (its content is genuinely there), so
        # the cross-bullet dedup never silently downgrades a grounded bullet to ungrounded.
        bullets: list[Bullet] = []
        for pb in pending:
            blk_objs, cite = pb["blocks"], pb["cite"]
            for b in blk_objs:
                kept_refs.add(b.frame_ref)
            folded_into = next((fold_target[r] for r in pb["origRefs"] if r in fold_target), None)
            if blk_objs:
                primary_ref = blk_objs[0].frame_ref
                anchor = Anchor(kind="frame", timestamp_sec=float(suggested[primary_ref]),
                                frame_ref=primary_ref)
            elif folded_into is not None:
                anchor = Anchor(kind="frame", timestamp_sec=float(suggested[folded_into]),
                                frame_ref=folded_into)
            elif cite is not None and transcript_segments:
                seg = _snap_segment(float(cite), transcript_segments)
                anchor = Anchor(kind="transcript", timestamp_sec=float(seg["start"]),
                                transcript_text=_span_text(float(seg["start"]), transcript_segments))
            else:
                anchor = None  # genuinely ungrounded -> shown, not clickable
            bullets.append(Bullet(text=pb["text"], anchor=anchor, blocks=blk_objs))

        if not bullets:
            rep.issues.append(f"section {sid}: no bullets; section skipped")
            continue

        sections.append(Section(
            id=sid,
            title=(sp.get("title") or sid).strip(),
            gist=(sp.get("gist") or "").strip(),
            bullets=bullets,
        ))
        rep.bullets_per_section[sid] = len(bullets)

    # One frames registry from the manifest; enrich the frames that carry a real (kept) block.
    frames = []
    for f in manifest_frames:
        fid = f["id"]
        placed = fid in kept_refs
        frames.append(Frame(
            id=fid,
            timestamp_sec=float(f["timestampSec"]),
            image_path=f["imagePath"],
            ocr_text=ocr_by_id.get(fid) if placed else None,
            vision_description=(
                f"On-screen {blocks_by_ref[fid]['type']} extracted by Claude vision (model {model})."
                if placed else None
            ),
        ))

    note = Note(
        video=Video(
            id=video_meta["id"], title=video_meta["title"], url=video_meta["url"],
            duration_sec=int(round(float(video_meta["durationSec"]))),
        ),
        frames=frames,
        sections=sections,
    )

    # Report: placement + grounding tiers (reuse the Step 2/5A checker as the source of truth).
    # all_block_refs partition into: placed (kept) | deduped (claimed then dropped) | unplaced.
    rep.section_count = len(sections)
    rep.placed_block_refs = sorted(kept_refs)
    rep.deduped_block_refs = sorted(used_refs - kept_refs)
    rep.unplaced_block_refs = [r for r in all_block_refs if r not in used_refs]
    rep.unplaced_reasons = {
        r: (plan.get("unplacedReasons", {}) or {}).get(r, "left unplaced by the planner")
        for r in rep.unplaced_block_refs
    }
    g = check_note_grounding(note)
    rep.bullets_by_tier = {
        "visual": g.visual_bullets, "spoken": g.spoken_bullets, "ungrounded": g.ungrounded_bullets,
    }
    return note, rep
