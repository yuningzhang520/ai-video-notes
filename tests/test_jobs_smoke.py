"""Smoke test for the jobs API over the Step 5A bullet schema.

Proves the async contract end to end against the in-memory store: POST creates a job, an
immediate poll observes ``processing``, status flips to ``done``, and the result validates
against the Note schema with bullets + the grounding contract wired up.
"""

import time

from fastapi.testclient import TestClient

from app.grounding import bullet_is_grounded, frame_ids
from app.jobs import PROCESSING_DELAY_SEC
from app.main import app
from app.models import Note

client = TestClient(app)


def test_job_lifecycle_processing_to_done_then_result():
    created = client.post("/api/jobs", json={"url": "https://www.youtube.com/watch?v=GDm_uH6VxPY"})
    assert created.status_code == 200
    job_id = created.json()["job_id"]
    assert job_id

    # Immediately observable as processing; result not available yet.
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "processing"
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409

    # After the delay it flips to done.
    time.sleep(PROCESSING_DELAY_SEC + 0.2)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"

    # Result validates, and carries the grounding contract: >=1 section, each with >=1 bullet,
    # every block resolves, and at least one grounded (verifiable) bullet exists.
    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    note = Note.model_validate(result.json())

    assert note.sections, "expected at least one section"
    ids = frame_ids(note)
    grounded_total = 0
    for sec in note.sections:
        assert sec.bullets, f"section {sec.id} has no bullets"
        for b in sec.bullets:
            if bullet_is_grounded(b, ids):
                grounded_total += 1
            for block in b.blocks:
                assert block.frame_ref in ids
    assert grounded_total >= 1, "expected at least one grounded bullet"


def test_unknown_job_is_404():
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_result_uses_camelcase_wire_format():
    job_id = client.post("/api/jobs", json={"url": "x"}).json()["job_id"]
    time.sleep(PROCESSING_DELAY_SEC + 0.2)
    body = client.get(f"/api/jobs/{job_id}/result").json()
    # Wire format is camelCase, as the frontend consumes it.
    assert "durationSec" in body["video"]
    assert body["sections"][0]["bullets"][0]["text"]
    assert body["frames"][0]["imagePath"].startswith("/static/frames/")
