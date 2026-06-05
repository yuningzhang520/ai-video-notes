"""Shared deterministic text-overlap primitives (pure; no I/O, no model).

One home for the line-normalization + overlap rules used in two places:
  - Step 5A in-bullet block dedup (``app/structure.py``): collapse a near-superset
    visual block into its fuller twin (a slide revealed line-by-line);
  - Step 6 frame selection (``app/frame_select.py``): fold a partial reveal frame
    into its fuller settled state across adjacent representatives.

Both callers use byte-identical normalization/overlap so the two dedup passes agree.
The bodies here are lifted verbatim from the original ``app/structure.py`` helpers --
this is a pure move, not a semantics change.
"""

from __future__ import annotations

import difflib
import re


def norm_lines(content: str) -> list[str]:
    """Normalize text for overlap comparison: non-empty, stripped, lowercased lines."""
    return [ln.strip().lower() for ln in (content or "").splitlines() if ln.strip()]


def flatten(content: str) -> str:
    """Fence-stripped, indentation/whitespace-normalized, lowercased single string.

    For similarity-ratio and prefix tests that should IGNORE indentation, wrapping, and the noisy
    spacing OCR introduces -- so two captures of the same editor screen read as the same string."""
    body = "\n".join(ln for ln in (content or "").splitlines() if not ln.lstrip().startswith("```"))
    lines = [re.sub(r"\s+", " ", ln.strip()).lower() for ln in body.splitlines()]
    return " ".join(ln for ln in lines if ln)


def text_prefix_subset(small: str, full: str) -> bool:
    """True iff `small`'s flattened content is a literal prefix of `full`'s -- a typewriter/reveal
    PARTIAL that carries no information beyond a leading slice of `full` (0 unique content)."""
    fs, ff = flatten(small), flatten(full)
    return len(fs) >= 10 and ff.startswith(fs)


def similarity_ratio(a: str, b: str) -> float:
    """Deterministic difflib character-similarity on the flattened (indentation-normalized) text.
    1.0 == identical after normalization; high == same screen re-captured a few chars apart."""
    return difflib.SequenceMatcher(None, flatten(a), flatten(b)).ratio()


def symmetric_ratio(a: str, b: str) -> float:
    """Order-INDEPENDENT difflib similarity: the min over both argument orders. difflib's ratio()
    is mildly asymmetric (the autojunk heuristic treats the 2nd sequence specially), so the raw
    value can flip a fold decision based on compare order. Taking the min makes it stable and
    conservative -- it folds only if BOTH directions agree the texts are similar."""
    return min(similarity_ratio(a, b), similarity_ratio(b, a))


def content_tokens(content: str) -> set[str]:
    """The alphanumeric token SET of the content, code fences dropped. Markdown markers (#, *, |)
    and all punctuation fall out of the alnum extraction, so framing/formatting can't pollute it."""
    body = "\n".join(ln for ln in (content or "").splitlines() if not ln.lstrip().startswith("```"))
    return set(re.findall(r"[a-z0-9]+", body.lower()))


def token_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the two contents' token sets -- ORDER-INDEPENDENT and immune to the
    framing/description wording that pollutes character-difflib (those words are a shared minority
    that cancels in a set overlap). 1.0 == identical vocabulary; the on-screen content dominates."""
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def line_overlap(a: str, b: str) -> float:
    """Shared distinct lines / lines(shorter). 1.0 means one is a line-superset of the other."""
    la, lb = set(norm_lines(a)), set(norm_lines(b))
    if not la or not lb:
        return 0.0
    return len(la & lb) / min(len(la), len(lb))


def more_complete_text(a: str, b: str) -> bool:
    """True iff `a` is strictly more complete than `b` (more lines, then longer normalized text)."""
    na, nb = norm_lines(a), norm_lines(b)
    if len(na) != len(nb):
        return len(na) > len(nb)
    return len(" ".join(na)) > len(" ".join(nb))
