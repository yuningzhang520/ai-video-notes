"""In-memory job store for the Step 1A skeleton.

No queue, no worker, no persistence -- jobs live in a dict for the life of the
process. Status is DERIVED from a clock (``ready_at``) rather than mutated by a
background task: this keeps processing -> done observable while staying fully
deterministic and race-free to test. The fixture "work" is instant; the small
delay only exists to make the async contract visible.
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .fixture import DEMO_VIDEO_ID, build_fixture_note
from .models import JobStatus, Note

# Seconds a job reports "processing" before flipping to "done". Long enough that
# an immediate poll observes "processing"; short enough not to slow the UI.
PROCESSING_DELAY_SEC = 0.6

# Step 4C: the committed demo Note -- the FULL multi-section, topic-segmented note built
# from the transcript + all 11 extracted VisualBlocks. The deployed host can't fetch YouTube,
# so for the demo video we serve this pre-built artifact instead of the fixture. NO
# request-time vision or structuring -- it's read off disk.
_DEMO_NOTE_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / DEMO_VIDEO_ID / "note_full.json"
_demo_note_cache: Note | None | bool = None  # None=unloaded, False=tried+absent, Note=loaded


def _load_demo_note() -> Note | None:
    """Load + cache the committed demo Note. Returns None if it isn't present/valid
    (e.g. before scripts/build_full_note.py has run), so callers fall back to the fixture."""
    global _demo_note_cache
    if _demo_note_cache is None:
        try:
            _demo_note_cache = Note.model_validate_json(_DEMO_NOTE_PATH.read_text())
        except Exception:
            _demo_note_cache = False
    return _demo_note_cache or None


def _note_for(url: str) -> Note:
    """The demo video gets the committed real-block Note; everything else gets the fixture."""
    if DEMO_VIDEO_ID in (url or ""):
        demo = _load_demo_note()
        if demo is not None:
            return demo
    return build_fixture_note()


@dataclass
class Job:
    id: str
    url: str
    ready_at: float                       # monotonic deadline
    note: Note = field(default=None)      # the fixture result

    @property
    def status(self) -> JobStatus:
        return "done" if time.monotonic() >= self.ready_at else "processing"


class JobStore:
    """Process-lifetime, in-memory map of job id -> Job."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, url: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            url=url,
            ready_at=time.monotonic() + PROCESSING_DELAY_SEC,
            note=_note_for(url),
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


# Single shared store for the run.
store = JobStore()
