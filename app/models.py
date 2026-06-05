"""Pydantic models for the Note schema (mirrors APPROACH.md).

Wire format is camelCase: Python fields are snake_case and an alias generator
maps them to camelCase. FastAPI serializes responses by alias by default, so the
JSON the frontend reads uses keys like ``frameRef`` / ``timestampSec`` directly.

The Note schema adds a top-level ``frames`` registry to APPROACH.md's
``Note { video, sections }`` so that every ``frameRef`` (on an Anchor or a
VisualBlock) can resolve to a concrete Frame. Without it the grounding pointer
has nowhere to point.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

VisualBlockType = Literal["code", "table", "diagram", "text"]
Confidence = Literal["low", "med", "high"]
JobStatus = Literal["processing", "done", "error"]
AnchorKind = Literal["frame", "transcript"]


class _Base(BaseModel):
    # camelCase on the wire, but also accept snake_case when constructing in Python.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Video(_Base):
    id: str
    title: str
    url: str
    duration_sec: int


class Frame(_Base):
    id: str
    timestamp_sec: float
    image_path: str
    ocr_text: Optional[str] = None          # from the gate (best-effort)
    vision_description: Optional[str] = None  # only if vision ran on the frame


class VisualBlock(_Base):
    type: VisualBlockType
    content: str                  # extracted artifact as markdown
    frame_ref: str                # -> Frame.id  (PROVENANCE)
    confidence: Optional[Confidence] = None  # self-reported triage signal, NOT calibrated


class Anchor(_Base):
    """A bullet's verifiable SOURCE pointer. Two grounded kinds (APPROACH.md's
    "frameRef OR transcript span", now both implemented):

      - kind="frame": grounded on an on-screen artifact -> ``frame_ref`` resolves to a
        Frame and ``timestamp_sec`` is the verify-lead; the source frame is shown on verify.
      - kind="transcript": grounded on a spoken moment -> ``transcript_text`` holds the
        caption span and ``timestamp_sec`` is the (snapped) segment start; verify seeks
        the player there, no frame.

    A bullet with ``anchor=None`` is genuinely ungrounded (shown, never click-to-verify).
    The claim itself lives in ``Bullet.text``, so there is no ``claim_ref`` here.
    """

    kind: AnchorKind = "frame"
    timestamp_sec: float                    # seek target: frame verify-lead OR segment start
    frame_ref: str = ""                     # -> Frame.id, when kind="frame"
    transcript_text: Optional[str] = None   # caption span shown, when kind="transcript"


class Bullet(_Base):
    """One self-contained logical point -- the primary content of a Section.

    ``text`` is the LLM-synthesized point (the claim). ``anchor`` is its verifiable source
    (None = ungrounded). ``blocks`` are the extracted artifacts supporting THIS point,
    carried VERBATIM from vision (the model never rewrites block content into ``text``);
    0, 1, or several.
    """

    text: str
    anchor: Optional[Anchor] = None
    blocks: list[VisualBlock] = []


class Section(_Base):
    id: str
    title: str                    # logical topic, LLM-derived
    gist: str                     # 1-2 sentence spoken summary (transcript)
    bullets: list[Bullet] = []    # the topic's substance; each a self-contained, grounded point


class Note(_Base):
    video: Video
    frames: list[Frame] = []      # registry that frameRef resolves into
    sections: list[Section] = []
