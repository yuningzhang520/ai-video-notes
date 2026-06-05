"""The canned demo Note returned for non-demo URLs (and the deterministic test baseline).

This is FIXTURE data. No video has been downloaded and no frames have been extracted -- the
frame images under static/frames/ are labeled placeholders at real timestamps within the demo
video. The real pipeline (download / dedup / gate / vision / structure) populates this same
schema for real for the demo URL.

It is hand-authored to exercise ALL THREE grounding tiers of the Step 5A bullet model, so the
grounding tests have a controllable, self-contained baseline:
  - VISUAL bullet     -- anchor.kind="frame", with a supporting VisualBlock (3 of these);
  - SPOKEN bullet     -- anchor.kind="transcript", a caption span + timestamp, no frame (1);
  - UNGROUNDED bullet -- anchor=None, shown but not click-to-verify (1; the Step 2 demo).
"""

from .models import Anchor, Bullet, Frame, Note, Section, Video, VisualBlock

DEMO_VIDEO_ID = "GDm_uH6VxPY"
DEMO_VIDEO_URL = "https://www.youtube.com/watch?v=GDm_uH6VxPY"


def build_fixture_note() -> Note:
    return Note(
        video=Video(
            id=DEMO_VIDEO_ID,
            title="AI agent design patterns",
            url=DEMO_VIDEO_URL,
            # Placeholder duration; the real value is read from metadata in a later step.
            duration_sec=1500,
        ),
        frames=[
            Frame(
                id="frame-0001",
                timestamp_sec=42,
                image_path="/static/frames/frame-0001.jpg",
                ocr_text="REASON -> ACT -> OBSERVE",
                vision_description=(
                    "A circular flow diagram with three nodes -- Reason, Act, Observe -- "
                    "connected in a loop, with an arrow exiting to a Done state."
                ),
            ),
            Frame(
                id="frame-0002",
                timestamp_sec=96,
                image_path="/static/frames/frame-0002.jpg",
                ocr_text='def get_weather(location: str, unit: str = "celsius") -> str:',
                vision_description=(
                    "A code editor showing a Python function definition for a weather "
                    "tool, including a docstring describing its purpose."
                ),
            ),
            Frame(
                id="frame-0003",
                timestamp_sec=158,
                image_path="/static/frames/frame-0003.jpg",
                ocr_text="Pattern | Best for | Cost   Single agent ...   Multi-agent ...",
                vision_description=(
                    "A two-column comparison table contrasting single-agent and "
                    "multi-agent designs across best-for and cost rows."
                ),
            ),
        ],
        sections=[
            Section(
                id="sec-1",
                title="The agent loop: reason, act, observe",
                gist=(
                    "An AI agent isn't a single prompt -- it runs a loop: reason about the "
                    "goal, take an action with a tool, observe the result, and repeat."
                ),
                bullets=[
                    # VISUAL bullet: grounded on the loop diagram.
                    Bullet(
                        text=(
                            "An agent runs a reason-act-observe loop: it reasons about the "
                            "current state, chooses an action (usually a tool call), executes "
                            "it, and feeds the observation into the next reasoning step, "
                            "repeating until it decides the goal is met."
                        ),
                        anchor=Anchor(kind="frame", timestamp_sec=42, frame_ref="frame-0001"),
                        blocks=[
                            VisualBlock(
                                type="diagram",
                                content="Reason -> Act -> Observe -> (loop back to Reason) -> Done",
                                frame_ref="frame-0001",
                                confidence="high",
                            )
                        ],
                    ),
                    # SPOKEN bullet: grounded on a transcript moment, no frame.
                    Bullet(
                        text=(
                            "The speaker frames every later pattern in the talk as a variation "
                            "on this one core loop, not as a separate mechanism."
                        ),
                        anchor=Anchor(
                            kind="transcript",
                            timestamp_sec=30,
                            transcript_text=(
                                "Every pattern we'll look at today is really just a variation "
                                "on this same reason, act, observe loop."
                            ),
                        ),
                    ),
                ],
            ),
            Section(
                id="sec-2",
                title="Defining tools the model can call",
                gist=(
                    "Tools are how the agent affects the world. Each is declared with a name, "
                    "a description, and a typed parameter schema the model fills in."
                ),
                bullets=[
                    # VISUAL bullet: grounded on the code frame.
                    Bullet(
                        text=(
                            "Each tool is declared with a name, a natural-language description "
                            "the model uses to decide when to call it, and a typed parameter "
                            "schema; the extracted snippet is best-effort OCR with the source "
                            "frame as ground truth."
                        ),
                        anchor=Anchor(kind="frame", timestamp_sec=96, frame_ref="frame-0002"),
                        blocks=[
                            VisualBlock(
                                type="code",
                                content=(
                                    "```python\n"
                                    'def get_weather(location: str, unit: str = "celsius") -> str:\n'
                                    '    """Look up the current weather for a location."""\n'
                                    "    ...\n"
                                    "```"
                                ),
                                frame_ref="frame-0002",
                                confidence="med",
                            )
                        ],
                    ),
                    # UNGROUNDED bullet (Step 2 demo): a spoken-only claim with NO traceable
                    # source. anchor=None resolves to nothing, so the reader shows it but marks
                    # it "no source" and makes it non-clickable -- never silently trusted.
                    Bullet(
                        text="The tool's result is fed back into the model's next reasoning step.",
                        anchor=None,
                    ),
                ],
            ),
            Section(
                id="sec-3",
                title="Single-agent vs. multi-agent patterns",
                gist=(
                    "Not every problem needs a swarm. The talk compares a single tool-using "
                    "agent against a multi-agent setup, and when each is the right call."
                ),
                bullets=[
                    # VISUAL bullet: grounded on the comparison table.
                    Bullet(
                        text=(
                            "Multi-agent shines when subtasks are independent and "
                            "parallelizable, but it adds coordination overhead you don't want "
                            "for a linear task -- this block is low-confidence, so verify it "
                            "against the dense source table."
                        ),
                        anchor=Anchor(kind="frame", timestamp_sec=158, frame_ref="frame-0003"),
                        blocks=[
                            VisualBlock(
                                type="table",
                                content=(
                                    "| Pattern | Best for | Cost |\n"
                                    "| --- | --- | --- |\n"
                                    "| Single agent | Linear, tool-driven tasks | Low latency |\n"
                                    "| Multi-agent | Independent, parallel subtasks | Coordination overhead |"
                                ),
                                frame_ref="frame-0003",
                                confidence="low",
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
