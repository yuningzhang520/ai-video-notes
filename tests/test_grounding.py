"""Step 2 + 5A: grounding-contract tests over the bullet model.

The fixture ships all three tiers by design -- 3 visual bullets, 1 spoken, 1 ungrounded
(the Step 2 shown-but-not-clickable demo). These pin the tier classification, the negative
cases, and that the ungrounded bullet still ships in the API payload (shown, not trusted).
"""

import time

from fastapi.testclient import TestClient

from app.fixture import build_fixture_note
from app.grounding import bullet_is_grounded, bullet_tier, check_note_grounding, frame_ids
from app.jobs import PROCESSING_DELAY_SEC
from app.main import app
from app.models import Anchor, Bullet, Note

client = TestClient(app)


def test_fixture_grounding_tiers():
    """The fixture classifies as designed: 3 visual, 1 spoken, 1 ungrounded, all blocks resolve."""
    rep = check_note_grounding(build_fixture_note())
    assert rep.visual_bullets == 3, rep.issues
    assert rep.spoken_bullets == 1, rep.issues
    assert rep.ungrounded_bullets == 1, rep.issues
    assert rep.verifiable_blocks == 3, rep.issues
    assert rep.unverifiable_blocks == 0, rep.issues


def test_bullet_tier_classification():
    """Every tier + every way to fall back to ungrounded."""
    ids = frame_ids(build_fixture_note())  # {frame-0001, frame-0002, frame-0003}

    def tier(anchor, text="a claim"):
        return bullet_tier(Bullet(text=text, anchor=anchor), ids)

    # visual: a frame anchor that resolves.
    assert tier(Anchor(kind="frame", timestamp_sec=42, frame_ref="frame-0001")) == "visual"
    # spoken: a transcript anchor with caption text.
    assert tier(Anchor(kind="transcript", timestamp_sec=30, transcript_text="they said it")) == "spoken"
    # ungrounded fallbacks:
    assert tier(None) == "ungrounded"                                                   # no anchor
    assert tier(Anchor(kind="frame", timestamp_sec=42, frame_ref="frame-nope")) == "ungrounded"  # dangling
    assert tier(Anchor(kind="frame", timestamp_sec=42, frame_ref="")) == "ungrounded"   # empty ref
    assert tier(Anchor(kind="transcript", timestamp_sec=30, transcript_text="  ")) == "ungrounded"  # no text
    assert tier(Anchor(kind="frame", timestamp_sec=42, frame_ref="frame-0001"), text="  ") == "ungrounded"  # empty claim

    good = Bullet(text="x", anchor=Anchor(kind="frame", timestamp_sec=42, frame_ref="frame-0001"))
    assert bullet_is_grounded(good, ids) is True
    assert bullet_is_grounded(Bullet(text="x", anchor=None), ids) is False


def test_api_result_ships_the_ungrounded_bullet():
    """The non-demo (fixture) result validates AND still contains the ungrounded bullet,
    so the show-but-mark behavior stays exercisable in the app."""
    job_id = client.post("/api/jobs", json={"url": "x"}).json()["job_id"]
    time.sleep(PROCESSING_DELAY_SEC + 0.2)
    body = client.get(f"/api/jobs/{job_id}/result").json()

    note = Note.model_validate(body)  # still schema-valid
    assert check_note_grounding(note).ungrounded_bullets == 1

    bullets = [b for s in body["sections"] for b in s["bullets"]]
    ungrounded = [b for b in bullets if b["anchor"] is None]
    assert len(ungrounded) == 1
    assert ungrounded[0]["text"]  # still has the claim text -> shown, just not clickable
