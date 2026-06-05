"""Grounding-integrity checks for a Note (Step 2, extended in Step 5A).

The grounding contract (APPROACH.md / CLAUDE.md): a claim renders as grounded and
click-to-verify ONLY if it carries a traceable, resolvable source. The unit is now the
BULLET (Step 5A) -- each bullet is one claim, and falls into exactly one of three tiers:

  - "visual"     -- anchor.kind="frame": frame_ref resolves to a Frame in Note.frames, with
                    a timestamp (the verify-lead). Verify shows that frame.
  - "spoken"     -- anchor.kind="transcript": a transcript timestamp + the caption text of
                    that span (no frame). Verify seeks the player to the spoken moment.
  - "ungrounded" -- anchor is None (or fails its kind's checks). Shown, never trusted/clickable.

Visual and spoken are EQUAL-status grounded tiers. Anything else is ungrounded -- shown, but
never silently trusted. The frontend makes the same decision independently and defensively;
this module is the backend's source of truth for tests and server-side integrity checks.
"""

from dataclasses import dataclass, field

from .models import Bullet, Note, VisualBlock

# A bullet's grounding tier: "visual" | "spoken" | "ungrounded".


def frame_ids(note: Note) -> set[str]:
    """The set of Frame ids a frameRef can resolve to."""
    return {f.id for f in note.frames}


def bullet_tier(bullet: Bullet, ids: set[str]) -> str:
    """Classify a bullet as "visual", "spoken", or "ungrounded".

    A bullet with no text, no anchor, or an anchor that fails its kind's checks is
    ungrounded. Frame anchors must resolve into `ids`; transcript anchors must carry a
    non-empty caption span. Both grounded tiers require a timestamp to seek to.
    """
    a = bullet.anchor
    if a is None or not (bullet.text and bullet.text.strip()) or a.timestamp_sec is None:
        return "ungrounded"
    if a.kind == "frame":
        return "visual" if a.frame_ref in ids else "ungrounded"
    if a.kind == "transcript":
        return "spoken" if (a.transcript_text and a.transcript_text.strip()) else "ungrounded"
    return "ungrounded"


def bullet_is_grounded(bullet: Bullet, ids: set[str]) -> bool:
    """True iff the bullet is verifiable (visual or spoken)."""
    return bullet_tier(bullet, ids) != "ungrounded"


def block_is_verifiable(block: VisualBlock, ids: set[str]) -> bool:
    """True iff the block's frameRef is present and resolves into `ids`."""
    return bool(block.frame_ref) and block.frame_ref in ids


@dataclass
class GroundingReport:
    visual_bullets: int = 0
    spoken_bullets: int = 0
    ungrounded_bullets: int = 0
    verifiable_blocks: int = 0
    unverifiable_blocks: int = 0
    issues: list[str] = field(default_factory=list)


def check_note_grounding(note: Note) -> GroundingReport:
    """Classify every bullet (and its blocks) by grounding tier. Does not mutate the note;
    returns a report for tests / integrity checks. Ungrounded bullets are NOT an error --
    they are shown, just not clickable -- so callers decide what to do about issues.
    """
    ids = frame_ids(note)
    rep = GroundingReport()
    for sec in note.sections:
        for i, bullet in enumerate(sec.bullets):
            tier = bullet_tier(bullet, ids)
            if tier == "visual":
                rep.visual_bullets += 1
            elif tier == "spoken":
                rep.spoken_bullets += 1
            else:
                rep.ungrounded_bullets += 1
                rep.issues.append(f"section {sec.id} bullet {i}: ungrounded (no resolvable source)")
            for block in bullet.blocks:
                if block_is_verifiable(block, ids):
                    rep.verifiable_blocks += 1
                else:
                    rep.unverifiable_blocks += 1
                    rep.issues.append(
                        f"section {sec.id} bullet {i}: unverifiable block (frameRef={block.frame_ref!r})"
                    )
    return rep
