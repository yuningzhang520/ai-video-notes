# Visual Notes from Video

> Turn a visually-dense video — a conference talk, a coding tutorial, a lecture — into structured notes where the on-screen content (code, tables, diagrams) is extracted, and **every point links back to the exact frame it came from, so you can verify it.**

[![Watch the demo](https://img.youtube.com/vi/3rRKfdJwtw8/maxresdefault.jpg)](https://youtu.be/3rRKfdJwtw8)

*~5-minute walkthrough — click to watch.*

**Live demo:** https://luma-video-notes.onrender.com

---

## The problem

Most video summarizers are transcript-first. For a talking-head video that's fine — the audio carries the meaning. But the videos people actually take notes on — technical talks, tutorials, lectures — the screen carries what the speaker never says out loud: the exact code, the numbers in a table, the structure of a diagram. Transcript-only tools drop all of that silently, so the notes read complete while missing the one thing you opened the video for.

## What I built

A tool that does two things differently:

- **It spends vision where it counts.** Instead of summarizing the transcript, it extracts the on-screen content the audio doesn't carry — and only on the frames that actually have it.
- **It makes every point verifiable.** Each grounded point links to the exact frame (or transcript moment) it came from. Click it, and the embedded video seeks to that moment and shows the source — so you can catch the model when it's wrong, instead of just trusting a summary.

The result is a note you can skim, expand, and fact-check — not a wall of text you have to take on faith.

## Who it's for

Anyone who learns from dense video and needs the details to be right: engineers following a technical talk, students working through a lecture, anyone who takes notes on a tutorial and needs the actual code, not a paraphrase. The product is built around trust — you can always see the screen the model read.

## See it in action

The demo runs on a Google Cloud talk about AI agent design patterns — a slide-dense, code-heavy video that's exactly the kind transcript-only tools fail on.

- **Click-to-verify** — tap any point; the video seeks to that moment and the source frame appears.
- **Two grounding tiers** — visual points show the source frame; spoken points show the transcript moment. Nothing is faked: a point with no real source is shown but never presented as verified.
- **Confidence as triage** — the riskiest extractions are flagged so you check those first.

---

## How it works

A batch pipeline — process a video once, then read. In order:

1. **Ingest** — pull the transcript (captions) and densely sample frames.
2. **Select** — collapse the dense frames to one representative per on-screen state, deterministically (no model deciding what to look at).
3. **Gate** — a cheap on-screen-text check decides which frames are worth vision; talking-head frames the transcript already covers are skipped.
4. **Vision** — a vision model extracts the salient artifact (code, table, diagram, text) from each gated frame, one block per frame.
5. **Structure** — one LLM pass organizes everything into topic sections with grounded bullets; deterministic assembly builds the final note from the extracted blocks, carried verbatim.
6. **Render** — a hierarchical reader with click-to-verify against the embedded player.

## Architecture & key decisions

The full write-up — decisions, trade-offs, what breaks first, what I'd build next — is in **[APPROACH.md](APPROACH.md)**. The decisions I'm most proud of:

- **Explicit frame pipeline, not native video ingestion.** Handing the whole video to a multimodal model is simpler, but it loses frame-level provenance — the model can claim a timestamp I can't verify. Verifiable grounding is the whole point, so every extracted point carries the frame that produced it.

- **The model does the judgment; the plumbing stays deterministic.** Reading frames and structuring the note is the model's job. Deciding *which* frames to look at is not — I considered an LLM router and rejected it (an inference call to route costs about what it saves and adds a failure point). Routing runs on cheap, deterministic signals instead, which is also what keeps every point traceable.

- **A real-world deployment constraint, handled honestly.** Live YouTube fetch works locally but is blocked from a datacenter IP (a bot-wall). Rather than hide that behind proxies a reviewer couldn't reproduce, the deployed demo serves a real note pre-built from committed frames and transcript — and the constraint is documented as the first thing to solve for production.

- **Verification is the failure-mode design.** Vision will misread frames, so the design assumes it: every point is click-to-verify, the riskiest are flagged, and the source frame is always one click away as the ground truth.

## A concrete example of *why* this approach

On one frame, the OCR text gate read a class name as `Parallet-Agent` — garbled. The vision model read it correctly as `ParallelAgent`. Extracted text can be wrong, which is exactly why the source frame is always shown: the user is never blind to what the model actually saw.

## Built with

Python · FastAPI · Claude (vision + structuring) · a lightweight plain-HTML/CSS/JS reader with the embedded video player · ffmpeg / perceptual hashing / OCR for frame selection.

## Running it

```bash
# Docker (one command)
docker compose up        # then open http://localhost:8000

# Or locally
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

See **[APPROACH.md](APPROACH.md)** for the architecture deep-dive and **[PLAN.md](PLAN.md)** for the build log.