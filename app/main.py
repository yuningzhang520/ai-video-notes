"""FastAPI app: async in-memory jobs API + single-page reader.

Endpoints (Step 1A -- fixture data only):
  POST /api/jobs                 -> { job_id }      create a job from a YouTube URL
  GET  /api/jobs/{job_id}        -> { status }      processing | done | error
  GET  /api/jobs/{job_id}/result -> Note JSON       once status is done
  GET  /                         -> the reader (static/index.html)
  GET  /healthz                  -> { status: ok }  liveness probe (Step 1B)
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .fixture import DEMO_VIDEO_ID, DEMO_VIDEO_URL
from .ingest import DEFAULT_WORK_DIR, fetch_and_sample
from .jobs import store
from .models import JobStatus, Note
from .transcript import fetch_transcript

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Video -> Verifiable Visual Notes", version="0.1.0-1A")


class CreateJobRequest(BaseModel):
    url: str


class CreateJobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: JobStatus


@app.post("/api/jobs", response_model=CreateJobResponse)
def create_job(req: CreateJobRequest) -> CreateJobResponse:
    job = store.create(req.url)
    return CreateJobResponse(job_id=job.id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(status=job.status)


@app.get("/api/jobs/{job_id}/result", response_model=Note)
def get_job_result(job_id: str) -> Note:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job not done yet")
    return job.note


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe for platform health checks (added in Step 1B)."""
    return {"status": "ok"}


@app.get("/api/_probe/ingest")
def probe_ingest() -> dict:
    """INTERNAL / TEMPORARY diagnostic (Step 3A) -- NOT a product API.

    De-risks the highest-risk unknown: can the deployed host (Render, a
    datacenter IP) actually download a YouTube video and extract frames, or does
    YouTube throttle/bot-wall the cloud IP? Hitting this on the live URL runs the
    SAME code path as local, so we learn the answer FROM THE HOST.

    Fixed to the demo video ONLY -- it deliberately takes no URL param, so it can
    never become a public downloader. It does NOT touch POST /api/jobs or the
    fixture flow; the real result still comes from the fixture. fetch_and_sample
    is internally bounded (download + ffmpeg timeouts), so this request returns
    with ok=False on failure rather than hanging.
    """
    DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="probe-", dir=str(DEFAULT_WORK_DIR))
    result = fetch_and_sample(DEMO_VIDEO_URL, work_dir, every_sec=60, max_frames=3)
    return {"probe": "ingest", "demo_url": DEMO_VIDEO_URL, **result.to_dict()}


@app.get("/api/_probe/transcript")
def probe_transcript() -> dict:
    """INTERNAL / TEMPORARY diagnostic (Step 3A) -- NOT a product API.

    Checks the other half of the fallback: with host video download bot-walled,
    can we still pull the demo video's captions from Render's IP? Captions are an
    HTTP fetch from YouTube too, so the same datacenter-IP block can apply -- this
    probe tells us whether transcript is usable or we also need a committed
    transcript fixture. Fixed to the demo video; bounded; never crashes the app.
    Does NOT touch POST /api/jobs or the fixture flow.
    """
    result = fetch_transcript(DEMO_VIDEO_ID)
    sample = result.segments[:3]  # first few segments only -- a probe, not a dump
    return {
        "probe": "transcript",
        "video_id": DEMO_VIDEO_ID,
        "ok": result.ok,
        "segment_count": result.segment_count,
        "sample": sample,
        "error": result.error,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Serve the reader's assets (JS/CSS) and the frame images at /static/*.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
