# AGENTS.md — Video → Verifiable Visual Notes

Operational guardrails for this build: a tool that turns a visually-dense video (talk, coding tutorial) into topic-segmented notes where on-screen content (code / tables / diagrams) is extracted and every grounded bullet links to its source frame or transcript timestamp (click-to-verify). Full rationale -> **APPROACH.md** (read on demand); this file is the always-on rules. If the build diverges from APPROACH.md, update APPROACH.md to match.

## Non-negotiable design rules (do NOT relitigate or "improve" these)
- **Explicit frame pipeline, never native video ingestion.** Frame-level provenance is the whole point.
- **Deterministic frame selection. NO LLM router.** Change-aware selection: dense sample -> cropped-pHash segmentation -> one representative per segment; gate = on-screen-text presence. The model interprets selected frames; it does not decide which frames to inspect.
- **Vision runs only on gated frames.** At most one VisualBlock per frame (one salient artifact: code | table | diagram | text).
- **Grounding contract:** every grounded bullet carries a traceable source — a `frameRef` (visual) or transcript span (spoken). A bullet with no source renders inert (shown, not click-to-verify).
- **Confidence is triage, not truth.** Self-reported confidence (uncalibrated), surfaced as a triage flag. **NO LLM-as-judge at request time.**
- **Notes are topic-segmented & hierarchical** (gist -> expand), dual-stream (spoken gist + extracted visual specifics). Sections, not time-chunks.
- **Render is click-to-verify:** tapping a grounded bullet seeks the embedded player to the timestamp — a visual bullet shows its source frame, a spoken bullet the cited transcript text; ungrounded bullets render inert. Low-confidence blocks are flagged.
- **Re-prompt (if added later) re-runs only the structure step** (cheap, no re-vision); it re-works only what was already extracted. Not shipped as a request-time control in this slice.

## Out of scope — do NOT build (resist gold-plating)
- Single YouTube URL only. No uploads, no non-YouTube sources.
- No accounts, no persistence beyond in-memory job/session state.
- No real job queue / worker — jobs run in memory for the run.
- No multi-video / series merging.
- No perfect-code-OCR rabbit hole — extract best-effort, show the source frame as ground truth.
- No production concerns (caching / batching / dashboards / drift monitoring).
- Cap video length to stay single-pass — no long-video map-reduce.

## Stack & conventions
- **Backend:** Python + FastAPI — houses the build-time pipeline (ingest / selection / gate / vision / structure as separate, testable functions mirroring APPROACH.md); at runtime it serves the committed Note artifact + the frontend.
- **API: async jobs** — `POST /api/jobs` (start → job id) → `GET /api/jobs/{id}` (status) → `GET /api/jobs/{id}/result` (notes JSON); job state in memory for the run. Do NOT build a blocking endpoint or a real queue/worker.
- **Frontend:** a single-page reader in plain HTML/CSS/JS, embedding the YouTube IFrame Player API and calling `player.seekTo()` client-side on click. **Hard constraint: notes + player are co-resident in the client — verify must seek instantly, never round-trip the server.** (React/Vite is acceptable if preferred, but not required for a reader this size.)
- **Model:** Claude (Anthropic API) for **build-time** vision + structuring; key from `.env` (`ANTHROPIC_API_KEY`). The shipped runtime reads committed artifacts and does NOT call Anthropic.
- **Deploy:** a single containerized service (Docker; FastAPI serving API + static frontend). It needs `ffmpeg` / `yt-dlp` / Tesseract system packages, so it runs on a container host (Render / Railway / Fly), not serverless.

## Verify & hygiene
- **Verify before moving on:** each change should actually run end-to-end on a known test video (a smoke check), not just look right.
- Commit incrementally — small, legible commits as each piece lands.
- `.env` is gitignored — **never commit secrets.** Keep `.env.example` with `ANTHROPIC_API_KEY=`.
- Don't put any code in `./dist` — `submit.sh` wipes and recreates it on every run.

## Compaction
- When compacting context, preserve: the modified files, commands run, any failing tests, the next step, and any decision that diverged from APPROACH.md.
