"""Frame selection: deterministic, change-aware choice of which frames are worth vision.

This is the PLUMBING half of the pipeline (APPROACH.md steps 2-3): it decides which frames
are worth a (paid) vision call, using only cheap, deterministic signals -- NO LLM router, NO
vision, NO network. The model interprets frames; it does NOT choose them.

    dense sample  ->  cropped-pHash segments  ->  one representative per segment
                  ->  fold adjacent OCR-superset duplicates  ->  OCR/text gate  ->  candidates

Build-time only. ``scripts/create_demo_artifacts.py`` runs the SELECTION (dense 0.5 fps sample
-> ``segment`` on cropped-pHash jumps -> ``choose_representative`` per segment ->
``collapse_adjacent_reps``) and commits the representatives; ``scripts/analyze_demo_frames.py``
applies the text gate and commits ``frame_analysis.json``. The deployed app READS that JSON and
never runs OCR/segmentation at request time (the same posture as the committed vision result).

The decision logic is split from the image I/O so it is testable without the Tesseract binary:
``compute_phash`` / ``segmentation_phash`` / ``ocr_signals`` do the image work, while ``segment``
/ ``choose_representative`` / ``collapse_adjacent_reps`` are pure functions over hashes + OCR
text. See the change-aware block lower in this file for the SLIDE_REGION_FRAC /
SEGMENT_PHASH_THRESHOLD / ADJACENT_OVERLAP rationale.

Gate knobs (ground-truthed against the committed demo pack's contact sheet):

  - ``CONF_FLOOR = 40`` -- Tesseract per-word confidence below this is mostly JPEG-noise
    garbage; words at/above it are real on-screen text.
  - ``MIN_TEXT_SCORE = 8`` -- a representative needs at least this many confident words to be
    worth vision. Sparse-text title slides and label-light diagrams fall below it -- the
    documented text-gate limitation in APPROACH.md, not a bug: the source frame is always shown.

OCR uses pytesseract (the Tesseract engine). On a normal machine / in the Docker image
(``tesseract-ocr`` installed) it runs unmodified.
"""

from __future__ import annotations

from typing import Optional

import imagehash
import pytesseract
from PIL import Image

from .textsim import line_overlap, more_complete_text, norm_lines

# --- Deterministic gate knobs (see module docstring for the ground-truth rationale) --
CONF_FLOOR = 40               # Tesseract per-word confidence floor for a "real" word
MIN_TEXT_SCORE = 8            # confident-word count a representative needs to be a candidate


def phash_distance(a: str, b: str) -> int:
    """Hamming distance between two phash hex strings (popcount of the XOR).

    Equivalent to imagehash's ``hash_a - hash_b`` but operates on the hex strings, so the
    dedup pass is testable without decoding images.
    """
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def compute_phash(image: Image.Image) -> str:
    """Perceptual hash of a frame as a 16-hex-char string (deterministic; no model)."""
    return str(imagehash.phash(image))


def ocr_signals(image: Image.Image) -> tuple[str, int, int, int, float]:
    """Run Tesseract once and return ``(ocr_text, word_count, text_length, text_score,
    mean_conf)``.

    ``text_score`` is the count of CONFIDENT words (conf >= CONF_FLOOR) -- the gate signal,
    robust to the low-confidence noise JPEG compression sprinkles across a frame. The
    transcription rebuilds lines from Tesseract's (block, par, line) layout so ``ocr_text``
    reads roughly as laid out on screen (best-effort; the frame itself is ground truth).
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    confident = 0
    for i in range(len(data["text"])):
        token = (data["text"][i] or "").strip()
        if not token:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
        if conf >= CONF_FLOOR:
            confident += 1
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(token)

    ocr_text = "\n".join(" ".join(toks) for _, toks in sorted(lines.items()))
    word_count = sum(len(toks) for toks in lines.values())
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    return ocr_text, word_count, len(ocr_text), confident, mean_conf


# --- Change-aware selection (cropped-pHash segments + OCR-superset collapse) -----------------
# Whole-frame pHash is unusable on this source: a presenter-video panel on the right of every
# slide never stops moving, so even a frozen slide reads as Hamming 4-10 (ground-truthed in
# work/probe_sampling). The fix is to compute the SEGMENTATION distance on the slide region only
# -- crop the panel out. This crop conditions the distance signal ONLY; committed frames, OCR,
# vision, and verify all still use the full frame.
#
# Threshold = 12 is data-derived, NOT an over-segment bias. The single-agent Pros/Cons reveal
# steps by ~12 (a bullet appears); a slide CUT is >=16. A threshold of 12 keeps the whole reveal
# (title -> partial -> fully-revealed) inside ONE segment, so within-segment choose_representative
# picks the fullest frame by line COUNT (which survives OCR garble) -- this is the PRIMARY reveal
# fix. Lower thresholds split the reveal and the partial leaks through (the cross-segment collapse
# cannot fold it -- raw Tesseract garbles the same slide differently per frame, so line-overlap
# drops below the bar). Higher thresholds (>=14) start merging genuinely-distinct slides. The
# adjacent OCR-superset collapse is SECONDARY mop-up: at 0.85 it only folds near-identical
# duplicates (e.g. a held title slide sampled twice), not reveals.

SLIDE_REGION_FRAC = 0.75      # keep the left 75% (slide card); drop the right 25% (presenter box)
SEGMENT_PHASH_THRESHOLD = 12  # cropped-pHash jump that starts a new segment (keeps reveals whole)
ADJACENT_OVERLAP = 0.85       # adjacent reps with >= this OCR line-overlap fold to the fuller one


def crop_to_slide_region(image: Image.Image, frac: float = SLIDE_REGION_FRAC) -> Image.Image:
    """Crop to the left ``frac`` of the frame -- the slide region, excluding the presenter panel.
    Used ONLY to compute the segmentation pHash; never alters a committed / OCR'd / vision frame."""
    w, h = image.size
    return image.crop((0, 0, max(1, int(round(w * frac))), h))


def segmentation_phash(image: Image.Image, frac: float = SLIDE_REGION_FRAC) -> str:
    """Perceptual hash of the slide region only (panel cropped out) -- the change-aware distance signal."""
    return compute_phash(crop_to_slide_region(image, frac))


def segment(seg_phashes: list[str], threshold: int = SEGMENT_PHASH_THRESHOLD) -> list[int]:
    """Assign a segment id to each frame: a new segment starts when the cropped-pHash jump from
    the PREVIOUS frame exceeds ``threshold``. Pure (operates on hex hashes); frames MUST be
    timestamp-ordered. Held-static frames (<=2 once the panel is gone) chain into one segment; a
    slide cut (>=16) starts a new one. At the default threshold (12) a reveal step (~12) does NOT
    split, so the whole reveal stays in one segment and choose_representative picks the fullest
    frame -- this is what keeps the partial Pros/Cons state from leaking through as its own rep.
    """
    ids: list[int] = []
    cur = 0
    prev: Optional[str] = None
    for ph in seg_phashes:
        if prev is not None and phash_distance(prev, ph) > threshold:
            cur += 1
        ids.append(cur)
        prev = ph
    return ids


def choose_representative(frames: list[dict]) -> tuple[dict, str]:
    """Pick ONE representative for a segment's frames (timestamp-ordered).

    Primary -- OCR-superset: the most-complete frame by ``textsim.more_complete_text`` on
    ``ocrText`` (more lines, then longer text), ties -> later. Catches a fully-revealed slide
    (all bullets) over a partial one. Used only when the winner has >= 2 text lines -- there must
    be real text to be "more complete" about.

    Fallback (diagram / low-text, < 2 lines) -- the LAST frame of the segment: the last stable
    frame before the detected cut (the boundary jump is AFTER it, so it still shows this slide),
    and the latest/most-revealed. This is exactly "latter half, settled, tie -> later" once you
    note that an endpoint always minimises local motion; stating it as `frames[-1]` is honest
    rather than dressing it up as a neighbour-distance search that can only ever pick the endpoint.

    ``frames`` items need ``ocrText`` (plus whatever the caller carries through).
    """
    if len(frames) == 1:
        return frames[0], "sole frame in segment"

    best = frames[0]
    for f in frames[1:]:                          # tie -> later: replace unless `best` is strictly fuller
        if not more_complete_text(best["ocrText"], f["ocrText"]):
            best = f
    if len(norm_lines(best["ocrText"])) >= 2:
        return best, f"OCR-superset ({len(norm_lines(best['ocrText']))} lines, tie->later)"

    return frames[-1], "low-text/diagram: last settled frame before the cut"


def collapse_adjacent_reps(reps: list[dict], overlap_threshold: float = ADJACENT_OVERLAP
                           ) -> tuple[list[dict], list[dict]]:
    """Fold a partial reveal representative into its fuller settled twin.

    Cropped pHash cannot tell a reveal step from a slide cut, so a progressively-revealed slide
    over-splits into adjacent segments (partial, then full). This deterministic pass walks the
    reps in time order and, whenever an ADJACENT pair's ``ocrText`` line-overlap is >= the
    threshold (same slide, more revealed), keeps the more-complete one and drops the other -- the
    frame-level analogue of 5A's in-bullet dedup, run BEFORE the gate/vision so a partial never
    reaches the model. Only compares against the last kept rep, so distinct content in between
    breaks the chain. Returns ``(kept, drops)``; drops are ``{dropped, kept, overlap}`` (ids + ratio).
    """
    kept: list[dict] = []
    drops: list[dict] = []
    for r in reps:
        if kept:
            o = line_overlap(kept[-1]["ocrText"], r["ocrText"])
            if o >= overlap_threshold:
                if more_complete_text(r["ocrText"], kept[-1]["ocrText"]):
                    drops.append({"dropped": kept[-1]["id"], "kept": r["id"], "overlap": round(o, 3)})
                    kept[-1] = r
                else:
                    drops.append({"dropped": r["id"], "kept": kept[-1]["id"], "overlap": round(o, 3)})
                continue
        kept.append(r)
    return kept, drops
