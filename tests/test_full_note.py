"""Step 5A: full-note (bullet structure-pass) tests. NONE call Anthropic.

Three layers:
  - the committed note_full.json CONTRACT (validates; bullets in 4-6 logical sections; both
    grounded tiers present; blocks carried through UNCHANGED; one block in <=1 bullet; visual
    anchors use the verify-lead, spoken anchors snap to real transcript segment starts);
  - the PURE ``assemble_note`` rules with a synthetic plan (visual/spoken/ungrounded tiers,
    unknown ref dropped, a block placed once even if cited twice, junk block reported);
  - the jobs endpoint still serves the multi-section bullet note unchanged.
"""

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.grounding import bullet_tier, frame_ids
from app.jobs import PROCESSING_DELAY_SEC
from app.main import app
from app.models import Note, VisualBlock
from app.structure import (
    DEDUP_TEXT_OVERLAP,
    assemble_note,
    dedup_blocks_across_section,
    dedup_blocks_in_bullet,
)
from app.textsim import similarity_ratio, symmetric_ratio, text_prefix_subset, token_jaccard

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY"
NOTE_PATH = DATA_DIR / "note_full.json"

client = TestClient(app)


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


# --- Committed note_full.json contract -----------------------------------------------

def test_note_full_validates_and_is_well_formed():
    note = Note.model_validate_json(NOTE_PATH.read_text())
    ids = frame_ids(note)

    assert 4 <= len(note.sections) <= 6  # logical topics, not time-chunks
    seen_tiers = set()
    for sec in note.sections:
        assert sec.bullets, f"section {sec.id} has no bullets"
        for b in sec.bullets:
            seen_tiers.add(bullet_tier(b, ids))
            for blk in b.blocks:
                assert blk.frame_ref in ids
            a = b.anchor
            if a is None:
                continue
            if a.kind == "frame":
                assert a.frame_ref in ids
            elif a.kind == "transcript":
                assert a.transcript_text and a.transcript_text.strip()
                assert a.frame_ref == ""
    # Both grounded tiers are actually exercised by the real note.
    assert "visual" in seen_tiers and "spoken" in seen_tiers


def test_note_full_blocks_unchanged_and_placed_once():
    note = Note.model_validate_json(NOTE_PATH.read_text())
    src = {b["frameRef"]: b for b in _load("visual_blocks.json")["blocks"]}

    blocks = [blk for sec in note.sections for b in sec.bullets for blk in b.blocks]
    refs = [blk.frame_ref for blk in blocks]
    assert len(refs) == len(set(refs))  # one block in at most one bullet, across the whole note

    for blk in blocks:
        s = src[blk.frame_ref]
        assert blk.type == s["type"]
        assert blk.content == s["content"]      # byte-for-byte: the model never rewrote it
        assert blk.confidence == s["confidence"]


def test_note_full_anchors_use_verify_lead_and_real_segments():
    note = Note.model_validate_json(NOTE_PATH.read_text())
    fa = {f["id"]: f for f in _load("frame_analysis.json")["frames"]}
    seg_starts = {round(float(s["start"]), 3) for s in _load("transcript.json")["segments"]}

    visual = spoken = 0
    for sec in note.sections:
        for b in sec.bullets:
            a = b.anchor
            if a is None:
                continue
            if a.kind == "frame":
                visual += 1
                f = fa[a.frame_ref]
                assert a.timestamp_sec == f["suggestedVerifyTimestampSec"]
                assert 0 <= a.timestamp_sec <= f["timestampSec"]   # a non-negative lead
            elif a.kind == "transcript":
                spoken += 1
                assert round(a.timestamp_sec, 3) in seg_starts      # snapped to a real segment
                assert a.transcript_text.strip()
    assert visual >= 1 and spoken >= 1


# --- Pure assemble_note rules --------------------------------------------------------

def test_assemble_note_tiers_dedup_and_filter():
    manifest = _load("manifest.json")
    frame_analysis = _load("frame_analysis.json")
    visual_blocks = _load("visual_blocks.json")
    segments = _load("transcript.json")["segments"]
    seg_starts = {round(float(s["start"]), 3) for s in segments}
    suggested = {f["id"]: f["suggestedVerifyTimestampSec"] for f in frame_analysis["frames"]}
    src = {b["frameRef"]: b for b in visual_blocks["blocks"]}
    video_meta = {"id": "GDm_uH6VxPY", "title": "T", "url": "u", "durationSec": 500.75}

    plan = {
        "sections": [
            {"id": "s1", "title": "One", "gist": "g", "bullets": [
                {"text": "visual point", "blockRefs": ["frame-0019", "frame-9999"]},  # 9999 unknown
                {"text": "spoken point", "transcriptCiteSec": 61},
                {"text": "ungrounded point"},
                {"text": "dup grabs the used block", "blockRefs": ["frame-0019"]},     # already placed
            ]},
            {"id": "s2", "title": "Two", "gist": "g", "bullets": [
                {"text": "second visual", "blockRefs": ["frame-0056"]},
            ]},
        ],
        "unplacedBlockRefs": ["frame-0070"],
        "unplacedReasons": {"frame-0070": "junk: raw debug dump"},
    }

    note, rep = assemble_note(plan, manifest, frame_analysis, visual_blocks, segments, video_meta, "test-model")
    ids = frame_ids(note)
    s1, s2 = note.sections[0].bullets, note.sections[1].bullets

    # visual bullet: unknown ref dropped, block copied verbatim, anchor uses the verify-lead.
    assert bullet_tier(s1[0], ids) == "visual"
    assert [blk.frame_ref for blk in s1[0].blocks] == ["frame-0019"]
    assert s1[0].blocks[0].content == src["frame-0019"]["content"]
    assert s1[0].anchor.kind == "frame" and s1[0].anchor.timestamp_sec == suggested["frame-0019"]

    # spoken bullet: snapped to a real segment, caption text set, no blocks.
    assert bullet_tier(s1[1], ids) == "spoken"
    assert round(s1[1].anchor.timestamp_sec, 3) in seg_starts
    assert s1[1].anchor.transcript_text.strip() and not s1[1].blocks

    # ungrounded bullet: no anchor, text kept.
    assert bullet_tier(s1[2], ids) == "ungrounded" and s1[2].anchor is None and s1[2].text

    # duplicate blockRef: dropped (already placed) -> this bullet ends up ungrounded.
    assert not s1[3].blocks and bullet_tier(s1[3], ids) == "ungrounded"
    assert bullet_tier(s2[0], ids) == "visual"

    # one block in at most one bullet across the whole note.
    placed = [blk.frame_ref for sec in note.sections for b in sec.bullets for blk in b.blocks]
    assert placed.count("frame-0019") == 1
    assert sorted(placed) == ["frame-0019", "frame-0056"]

    assert rep.bullets_by_tier == {"visual": 2, "spoken": 1, "ungrounded": 2}
    assert "frame-0070" in rep.unplaced_block_refs
    assert rep.unplaced_reasons["frame-0070"].startswith("junk:")
    assert any("frame-9999" in it for it in rep.issues)       # unknown ref recorded
    assert any("already placed" in it for it in rep.issues)   # duplicate recorded


# --- Same-bullet text-overlap dedup --------------------------------------------------

def _vb(ref, lines, type="text"):
    return VisualBlock(type=type, content="\n".join(lines), frame_ref=ref, confidence="high")


def test_dedup_helper_superset_distinct_type_and_boundary():
    # Near-superset, same type -> drop the smaller, keep the fuller, record overlap (order-independent).
    small = _vb("f1", ["## Wrap up", "- a", "- b"])
    full = _vb("f2", ["## Wrap up", "- a", "- b", "- c"])
    for order in ([small, full], [full, small]):
        kept, drops = dedup_blocks_in_bullet(order)
        assert [k.frame_ref for k in kept] == ["f2"]
        assert drops == [{"dropped": "f1", "kept": "f2", "overlap": 1.0}]

    # Genuinely different same-type blocks (overlap 0.5) -> both kept.
    a = _vb("f3", ["line one", "line two", "line three", "line four"])
    b = _vb("f4", ["line one", "line two", "other x", "other y"])
    kept, drops = dedup_blocks_in_bullet([a, b])
    assert {k.frame_ref for k in kept} == {"f3", "f4"} and drops == []

    # Same lines but DIFFERENT type -> never merged.
    kept, drops = dedup_blocks_in_bullet([_vb("f5", ["x", "y", "z"], "text"),
                                          _vb("f6", ["x", "y", "z"], "code")])
    assert {k.frame_ref for k in kept} == {"f5", "f6"} and drops == []

    # Threshold boundary: overlap exactly DEDUP_TEXT_OVERLAP (0.75) -> treated as a duplicate;
    # just below (0.5) -> both kept.
    at = dedup_blocks_in_bullet([_vb("f7", ["a", "b", "c", "d"]), _vb("f8", ["a", "b", "c", "z"])])
    assert len(at[0]) == 1 and at[1][0]["overlap"] == DEDUP_TEXT_OVERLAP
    below = dedup_blocks_in_bullet([_vb("f9", ["a", "b", "c", "d"]), _vb("f10", ["a", "b", "y", "z"])])
    assert len(below[0]) == 2 and below[1] == []


# --- Section-scoped cross-bullet dedup (Step 6 follow-up) ---------------------------

def test_section_dedup_folds_code_subset_but_keeps_distinct_or_below_bar():
    full = _vb("f1", ["a=1", "b=2", "c=3", "d=4"], "code")
    subset = _vb("f2", ["a=1", "b=2"], "code")                 # true subset of f1
    dropped, drops = dedup_blocks_across_section([(0, full), (2, subset)], {})
    assert dropped == {"f2"}                                   # contained scroll-view folded
    assert drops[0]["dropped"] == "f2" and drops[0]["kept"] == "f1"
    # genuinely-distinct code (low ratio, no subset) -> keep both
    a = _vb("f3", ["x = fetch_weather(city)", "print(x.temperature)"], "code")
    b = _vb("f4", ["total = sum(prices)", "report(total, tax_rate)"], "code")
    assert dedup_blocks_across_section([(0, a), (1, b)], {}) == (set(), [])


def test_section_dedup_text_threshold_and_never_across_types():
    small, full = _vb("t1", ["x", "y"]), _vb("t2", ["x", "y", "z"])   # text overlap 1.0
    assert dedup_blocks_across_section([(0, small), (1, full)], {})[0] == {"t1"}
    # identical lines but different types never fold
    code, text = _vb("c1", ["x", "y", "z"], "code"), _vb("c2", ["x", "y", "z"], "text")
    assert dedup_blocks_across_section([(0, code), (1, text)], {})[0] == set()


def test_section_dedup_diagrams_fold_on_text_ratio():
    # diagrams fold on text-similarity ratio >= 0.75 (text-only path -- no image check on this path).
    d1 = _vb("g1", ["Trace panel", "agent_run museum 13s", "result: Academy of Sciences"], "diagram")
    d2 = _vb("g2", ["Trace panel", "agent_run museum 13s", "result: Academy of Sciences",
                    "search: best museums SF"], "diagram")
    far = {"g1": "0000000000000000", "g2": "ffffffffffffffff"}          # different images -> still folds
    assert dedup_blocks_across_section([(0, d1), (1, d2)], far)[0] == {"g1"}
    # genuinely different diagram text (low ratio) -> kept regardless of image
    e1 = _vb("h1", ["Sequential trace", "food agent then transport agent"], "diagram")
    e2 = _vb("h2", ["Parallel trace", "three specialists fire simultaneously",
                    "museum concert restaurant"], "diagram")
    assert dedup_blocks_across_section([(0, e1), (1, e2)], {})[0] == set()


def test_text_prefix_and_similarity_helpers():
    assert text_prefix_subset("Find a sushi place that's", "Find a sushi place that's open late")
    assert not text_prefix_subset("Find a sushi place that's open late", "Find a sushi place that's")
    assert not text_prefix_subset("totally different", "Find a sushi place that's open late")
    # similarity ignores indentation/wrapping; near-identical code scores high, distinct code low.
    assert similarity_ratio("```py\nx = f(a)\n  y = g(x)\n```", "```py\nx = f(a)\ny = g(x)\n```") > 0.95
    assert similarity_ratio("museum_finder = Agent(find museums)",
                            "concert_finder = Agent(find concerts)") < 0.85


def test_section_dedup_text_prefix_folds_typewriter_partial():
    # a typewriter PARTIAL (literal flattened prefix) folds into the fuller text, keeping the fuller.
    small = _vb("p1", ["Find a sushi place that's"])
    full = _vb("p2", ["Find a sushi place that's open late and nearby"])
    dropped, drops = dedup_blocks_across_section([(0, small), (1, full)], {})
    assert dropped == {"p1"} and drops[0]["kept"] == "p2"


def test_token_jaccard_and_symmetric_ratio_helpers():
    # token-Jaccard: order-independent, ignores markdown/framing; on-screen content dominates.
    assert token_jaccard("## Panel (left)\nJin Sho at 454 California Avenue",
                         "Result: Jin Sho at 454 California Avenue, CA") >= 0.6
    assert token_jaccard("the single agent uses one llm",
                         "parallel agents run three specialists") < 0.2
    assert token_jaccard("a b c", "c b a") == 1.0                 # order-independent
    # symmetric_ratio is order-independent (min of both difflib directions).
    a, b = "x = compute(a)\n  y = render(x)", "x = compute(a)\ny = render(x)\nz = 1"
    assert symmetric_ratio(a, b) == symmetric_ratio(b, a)
    assert symmetric_ratio(a, b) <= similarity_ratio(a, b)        # min <= either direction


def test_section_dedup_text_folds_high_ratio_recapture_but_keeps_distinct():
    # same text re-captured with minor OCR differences (ratio >= 0.85) folds to the fuller block.
    a = _vb("t1", ["Jin Sho is a sushi restaurant at 454 California Avenue, Palo Alto."])
    b = _vb("t2", ["Jin Sho is a sushi restaurant at 454 California Avenue, Palo Alto, CA 94306."])
    dropped, _ = dedup_blocks_across_section([(0, a), (1, b)], {})
    assert dropped == {"t1"}                                  # fuller kept, recapture folded
    # genuinely different text (low ratio, not a prefix/subset) -> both kept.
    c = _vb("t3", ["The single agent uses one LLM with a set of tools."])
    d = _vb("t4", ["Parallel agents run three independent specialists at once."])
    assert dedup_blocks_across_section([(0, c), (1, d)], {})[0] == set()


def test_section_dedup_code_ratio_folds_recapture_but_keeps_distinct():
    # same code re-captured a few chars on (NOT a clean line-subset) -> folds via the ratio path.
    a = _vb("c1", ["x = compute(a, b)", "y = render(x)", "return y"], "code")
    b = _vb("c2", ["x = compute(a, b)", "y = render(x)", "return yy"], "code")
    dropped, _ = dedup_blocks_across_section([(0, a), (1, b)], {})
    assert dropped == {"c1"}                                   # fuller (longer) kept, recapture folded
    # genuinely different code (low ratio, no subset) -> both kept.
    d1 = _vb("d1", ["weather = fetch_api(city)", "return weather.temperature"], "code")
    d2 = _vb("d2", ["total = sum(p.price for p in cart)", "apply_discount(total, promo)"], "code")
    assert dedup_blocks_across_section([(0, d1), (1, d2)], {})[0] == set()


def test_assemble_dedups_near_superset_blocks():
    """A bullet that references both single-agent code captures keeps only the fuller one,
    verbatim, with the anchor following the kept block -- and the drop is reported."""
    manifest, frame_analysis, visual_blocks = _load("manifest.json"), _load("frame_analysis.json"), _load("visual_blocks.json")
    segments = _load("transcript.json")["segments"]
    src = {b["frameRef"]: b for b in visual_blocks["blocks"]}
    suggested = {f["id"]: f["suggestedVerifyTimestampSec"] for f in frame_analysis["frames"]}
    video_meta = {"id": "GDm_uH6VxPY", "title": "T", "url": "u", "durationSec": 500.75}

    plan = {"sections": [{"id": "single", "title": "Single", "gist": "g", "bullets": [
        {"text": "the single-agent code", "blockRefs": ["frame-0019", "frame-0021"]}]}],
        "unplacedBlockRefs": [], "unplacedReasons": {}}

    note, rep = assemble_note(plan, manifest, frame_analysis, visual_blocks, segments, video_meta, "m")
    b = note.sections[0].bullets[0]

    assert [blk.frame_ref for blk in b.blocks] == ["frame-0019"]          # fuller kept, 0021 dropped
    assert b.blocks[0].content == src["frame-0019"]["content"]            # verbatim, never rewritten
    assert b.anchor.frame_ref == "frame-0019"                            # anchor follows the kept block
    assert b.anchor.timestamp_sec == suggested["frame-0019"]
    assert any(d["dropped"] == "frame-0021" and d["kept"] == "frame-0019" for d in rep.dedup_drops)
    assert "frame-0021" in rep.deduped_block_refs and "frame-0021" not in rep.placed_block_refs


def test_committed_note_dedups_root_agent_recaptures():
    """The plan references three root_agent captures (frame-0019/0020/0021 in one bullet);
    note_full.json carries only the fuller frame-0020 -- the within-bullet subset (0021) and the
    cross-bullet code-similarity recapture (0019) both fold in assembly, on the clean content."""
    plan = _load("section_plan.json")
    src = {b["frameRef"]: b for b in _load("visual_blocks.json")["blocks"]}
    note = Note.model_validate_json(NOTE_PATH.read_text())

    sa_plan = next(s for s in plan["sections"] if s["id"] == "single-agent-pattern")
    plan_refs = [r for bu in sa_plan["bullets"] for r in (bu.get("blockRefs") or [])]
    assert {"frame-0019", "frame-0020", "frame-0021"} <= set(plan_refs)   # plan: all captures present

    all_refs = [blk.frame_ref for s in note.sections for bu in s.bullets for blk in bu.blocks]
    assert "frame-0021" not in all_refs and "frame-0019" not in all_refs  # both recaptures folded out
    assert all_refs.count("frame-0020") == 1                              # only the fuller capture kept
    kept = next(blk for s in note.sections for bu in s.bullets for blk in bu.blocks
                if blk.frame_ref == "frame-0020")
    assert kept.content == src["frame-0020"]["content"]   # kept block is byte-identical (dropped, not rewritten)


# --- Endpoint shape unchanged, now serving the bullet note ---------------------------

def test_demo_job_serves_the_full_bullet_note():
    job_id = client.post("/api/jobs", json={"url": "https://www.youtube.com/watch?v=GDm_uH6VxPY"}).json()["job_id"]
    time.sleep(PROCESSING_DELAY_SEC + 0.2)
    body = client.get(f"/api/jobs/{job_id}/result").json()

    note = Note.model_validate(body)
    assert len(note.sections) >= 4
    assert "durationSec" in body["video"]
    assert body["frames"][0]["imagePath"].startswith("/static/demo/")
    # camelCase bullet shape the reader will consume in 5B.
    b0 = body["sections"][0]["bullets"][0]
    assert b0["text"]
    assert "anchor" in b0 and "blocks" in b0
