# Video → Verifiable Visual Notes

> This document records the architecture and decisions I committed to before writing code. Sections marked _(post-build)_ — live URL, run command, latency/cost numbers — are filled in once the system runs; anything that changes during the build is updated here to match what actually shipped.

## What I'm building & why this problem

A tool that turns a visually-dense video — a conference talk, a coding tutorial — into structured, navigable notes where the on-screen content (code, diagrams, slide data) is extracted and **every grounded bullet links back to its source frame (or transcript moment) and timestamp it came from**.

Today's video summarizers are transcript-first. For talking-head content that's fine — the audio carries the meaning. But for the videos people actually take notes on — lectures, tutorials, talks — the screen carries what the narrator never says out loud: the exact code, the numbers in a table, the structure of a diagram. Transcript-only tools drop that silently; the notes read complete while missing the thing you needed. My angle is to spend vision exactly where the screen carries content the audio doesn't, and to make the result verifiable so the user can catch the model when it's wrong. Visually-dense is what I optimize for, not a precondition I check: on a talking-head video the same pipeline finds little to extract and falls back to a transcript summary.

## How it works

Pipeline, in order:

1. **Ingest** — pull the transcript (YouTube captions) and densely sample frames at a fixed interval (~1 every 2s — a tunable knob, not a fixed law) as the base set the selector runs over.
2. **Select** — collapse the dense sample to one frame per on-screen state, deterministically. Whole-frame perceptual hashing is defeated here by a presenter-video panel that never stops moving, so segmentation runs on a *cropped* slide region: consecutive frames whose cropped pHash barely changes form one segment.

   Each segment contributes a single **representative** — the OCR-superset frame when a slide is being revealed, or the last stable frame when the segment is low-text or diagram-like. Adjacent representatives that are the same slide mid-reveal fold into the fuller one. A slide held for 60s collapses to one frame, not thirty; dense sampling catches fast reveals without making vision pay for every near-duplicate frame.

   A second, content-level dedup runs deterministically at assembly: within a bullet, and across a section, same-type blocks collapse when one's extracted text is a near-subset of the other. The fuller block is kept verbatim; the duplicate is dropped from the data, not hidden at render time.
3. **Gate** — cheap on-screen-text detection decides which surviving frames are worth vision. Text-heavy frames (where the transcript misses content) go to vision; plain talking-head frames are skipped — the transcript already covers them.
4. **Vision** — run a vision model only on the gated frames. Each yields at most one block — the salient on-screen artifact (code, table, diagram, text), extracted as a standalone unit and labeled with its dominant type. Resolution keys off that type, not frame density: **diagram** frames are re-extracted at 1920×1080 because a diagram's value is the image you read, while code/table/text frames stay at thumbnail resolution — their value is the extracted text, with the source frame shown on verify as ground truth.
5. **Structure** — one LLM pass turns the transcript + a compact summary of each extracted block into a **section plan** (topics, bullets, and the block or transcript span that grounds each bullet). Deterministic assembly then builds the final Note from that plan — copying the committed VisualBlocks **verbatim** and snapping each anchor to a real frame verify-lead or transcript segment — so the model never rewrites block content, and **every bullet must cite a frame or transcript span** (with a self-reported confidence per visual block).
6. **Render** — hierarchical view (skim -> expand) with click-to-verify: tapping a grounded bullet seeks the embedded video to that timestamp — a visual bullet then shows its source frame, a spoken bullet the cited transcript moment; ungrounded bullets render inert. Low-confidence blocks are flagged so the user checks those first.

Shape: a batch pipeline — process a video once, then read — not an interactive service, so per-request latency isn't the binding constraint (which is also why the re-pass can be deferred). The result is served to a lightweight web reader whose click-to-verify drives an embedded video player's seek.

Processing runs as an async job — POST a URL to start, poll for status, then GET the result — with job state held in memory for the run, so a slow multi-minute video doesn't hold a request open or trip a host timeout (a real queue/worker is what's-next). The service is containerized for its ffmpeg / yt-dlp / Tesseract system dependencies, so it deploys to a container host (Render / Railway / Fly), not serverless.

**Deployment constraint (measured on the live host, not assumed).** 

   Request-time YouTube fetch does not survive deployment. From the deployed host's datacenter IP (Render) both ingest paths are blocked by YouTube: the video download hits HTTP 429 + a "Sign in to confirm you're not a bot" wall (with ffmpeg and a JavaScript runtime present, and the android/ios/tv player clients tried — none cleared it), and `youtube-transcript-api` returns `RequestBlocked`, reporting that cloud-provider IPs are broadly blocked. Both paths work from a residential IP — the diagnostic ingest probe downloads the demo video and extracts three frames (a probe, distinct from the committed representative pack below); captions return 167 segments. 
  
   So the deployed demo does not depend on fetching at request time: real frames + transcript have now been pre-extracted from a residential IP and committed as a demo artifact pack — 93 representative frames selected from a 250-frame dense sample, the 167-segment transcript, and a manifest mapping frame ids to timestamps and served image paths — which the server reads reproducibly with no private credentials (cookies / browser-cookie auth / proxies are excluded as not reviewer-reproducible). 
  
   The full pipeline runs over this pack at build time — change-aware selection (dense sample → cropped-pHash segments → representatives) plus an OCR/text gate pick 36 vision candidates, Claude vision extracts one block per candidate, and a single structure pass plans the topic-segmented note that deterministic assembly builds — all committed, so the deployed demo serves a real multi-section, click-to-verify note with no request-time model calls. Reviewers who want to reproduce the AI build start from these committed frames, not a YouTube URL — re-running vision then structure with their own key (`build_visual_blocks.py --force` → `build_full_note.py --force`); fresh model output is schema-valid but not byte-identical, while the cached structure→note assembly stays deterministic (run commands in README.md). This changes only *where* ingest's inputs come from in the deployed demo — the explicit-frame pipeline above is unchanged. Internal diagnostic endpoints `/api/_probe/ingest` and `/api/_probe/transcript` recorded this.

Re-shaping the notes — length, emphasis, a topic that was extracted but dropped — would be cheap by design: it re-runs just the structure step, no re-vision. This slice ships no request-time re-prompt control, though — the structure pass is re-runnable on its own at build time, and a per-section re-prompt is left for what's next. It can't fix a frame the model misread (that's verify + the deferred re-pass) or content the gate never captured (a known limit); it only re-works what was already extracted.

## Key decisions & tradeoffs

**Explicit frame pipeline, not native video ingestion.** Handing the whole video to a multimodal model is simpler and does temporal reasoning for free, but it loses frame-level provenance: the model can *claim* a timestamp without my being able to verify where a claim came from, and it can hallucinate the timestamp too. My core feature is verifiable grounding, so I chose the explicit pipeline where every grounded bullet carries the frame or transcript span that produced it. Cost: more moving parts, and I'm rebuilding some sampling/temporal logic the native model does internally.

**The model does the judgment; the plumbing stays deterministic.** Reading code and diagrams off a frame, segmenting into topics, deciding what's worth a note, grounding each bullet — that's the model's work. Deciding which frames are worth vision is not: I considered an LLM router and rejected it, because an inference call to route costs roughly what it saves and adds a failure surface. Routing belongs to cheap signals — cropped-pHash segmentation + text-presence — with resolution as the only real knob. The cost is that a heuristic occasionally skips a frame that mattered or keeps a redundant one; acceptable, because the frame is always shown, so the user is never blind to what the model saw.

**Topic-segmented, hierarchical, dual-stream notes.** Sections are logical topics, not time-chunks — time-chunks aren't how anyone reviews notes. Each section is a one-line gist plus a list of **bullets** — self-contained logical points produced directly (not prose split into fragments), each individually verifiable, with any extracted visual specifics nested under the exact bullet they support: not "the speaker discussed an array" but the actual array from the screen, sitting under the point about it. A light fixed schema (sections of grounded bullets; visual blocks attach to the bullet they evidence) over fully-emergent structure — costs some natural fit, buys consistency and navigability.

**Verification is the failure-mode design.** Vision will misread frames, so the design assumes it rather than hiding it. Every grounded bullet is click-to-verify — jump to the timestamp, then see its source frame (visual) or the cited transcript moment (spoken); ungrounded bullets render inert. The riskiest blocks are flagged so attention goes there first. The flag is the model's self-reported confidence — uncalibrated, treated as triage, not truth. I deliberately don't run an LLM-as-judge at request time: a judge that can't see the frame can't catch the error that matters — the transcript doesn't contain the code — and one that re-reads the frame is just a second vision pass. So judging belongs in offline eval and re-reading belongs in the deferred re-pass; at runtime, the frame itself is the ground truth the user falls back to.

## Note schema

```
Note {
  video:    { id, title, url, durationSec }
  frames:   Frame[]      // registry every frameRef (Anchor/VisualBlock) resolves into
  sections: Section[]
}

Section {
  id
  title          // logical topic, LLM-derived
  gist           // 1-2 sentence spoken summary (transcript)
  bullets        // Bullet[] - the topic's substance; each a self-contained, verifiable point
}

Bullet {                // the primary content: a complete logical point, not prose
  text           // the LLM-synthesized bullet text (mechanism / example / why), produced directly
  anchor?        // its verifiable source; absent => genuinely ungrounded (shown, not clickable)
  blocks         // VisualBlock[] - extracted artifacts supporting THIS point; 0+, carried verbatim
}

VisualBlock {
  type           // dominant kind (extend as needed); sets display emphasis (diagram -> frame image, code/table -> extracted text), not body formatting
  content        // extracted artifact as markdown (code / table / text; renders per-part)
  frameRef       // -> Frame.id  (PROVENANCE)
  confidence?    // self-reported triage signal (low|med|high), NOT calibrated
}

Anchor {                // a bullet's source pointer; two grounded kinds (no claimRef - the claim is Bullet.text)
  kind           // "frame" | "transcript"
  timestampSec   // seek target: frame verify-lead, OR snapped transcript segment start
  frameRef       // -> Frame.id, when kind="frame" (image shown on verify)
  transcriptText?// the caption span quoted, when kind="transcript" (no frame; verify plays the spoken moment)
}

Frame {
  id
  timestampSec
  imagePath
  ocrText?            // from the gate
  visionDescription?  // only if vision ran on it
}
```

The grounding contract: every bullet must carry a traceable source — a `frameRef` (visual) **or** a transcript span (spoken) — or it renders ungrounded (shown, never click-to-verify). Both grounded tiers are equal status: clicking either seeks the co-resident player; a visual bullet also shows its source frame, a spoken bullet plays the cited moment.


## What I intentionally left out

- **Multi-video / series merging.** The schema keys notes by source, so this is a natural extension — but merging across a series (and de-duping overlapping content) is its own problem. Architected for, not built.
- **Upload + non-YouTube sources.** Single YouTube URL only. Captions-as-transcript is the cheapest path *where YouTube is reachable* — but, as the deployment-constraint note records, it is not reliable from a cloud host (cloud-IP block), which is why the deployed demo reads a committed transcript artifact (alongside committed pre-extracted frames) rather than fetching at request time. Supporting arbitrary uploads means owning transcription too.
- **Perfect code OCR.** Clean, correctly-indented code out of a screenshot is a rabbit hole. I extract best-effort and show the source frame alongside as ground truth, rather than over-invest in OCR accuracy.
- **Accounts, persistence, editing, export.** Out of scope for the slice.
- **Production concerns** — a real job queue/worker, application-level result caching, batching, dashboards, drift monitoring. Jobs run in-memory for the slice; the rest is listed under "what's next".

## What breaks first under pressure

_(Honest read of the weak points. Build-time vision cost was ≈ $0.21 for the 36 extracted blocks; the deployed server makes no model calls at request time, so request latency is cold-start / platform dependent.)_

- **Cloud YouTube fetch / captions are the first production pressure point.** Both the video download and the caption fetch are blocked from datacenter IPs (HTTP 429 / "confirm you're not a bot" wall / `RequestBlocked`), measured on the live host — so request-time ingestion does not survive deployment as-is. Production options — official API access, authenticated ingestion, residential-proxy infrastructure, or user upload — are all out of scope for this slice, which runs on committed pre-extracted demo artifacts instead — real frames + transcript, now generated and committed.
- **Long videos blow the context window.** Past a length threshold I'd need to chunk -> summarize per chunk -> merge, and the merge has to de-dupe sections or the same point repeats across chunk boundaries. This slice caps video length to stay single-pass; that cap is the first thing to break on a 2-hour talk.
- **The text-gate is a heuristic.** A frame with sparse but critical on-screen text (one key number, nothing else) can fall below the gate and skip vision. The dedup/gate thresholds are tuned, not learned.
- **Vision misreads dense frames.** Tightly-packed code or a busy diagram is where the vision model is most likely to be wrong — which is exactly why every visual block shows its source frame and carries a confidence flag.
- **Timestamp <-> frame alignment.** A sampled frame can sit at the very end of the interval its content was on screen — e.g. a code frame sampled around 6:11, right about where that code scrolls away — so a verify-seek to the raw sample timestamp can land just as the content leaves the screen, defeating the trust moment. The shipped mitigation is a small fixed lead: every anchor seeks to `suggestedVerifyTimestampSec = max(sampleTs - 3s, 0)` (computed deterministically, not by the model), landing a few seconds before the content leaves screen — a frame at 6:11 seeks to 6:08. It's a pragmatic mitigation, not a perfect one; a midpoint-of-on-screen-interval refinement (using the per-segment frame members now recorded) is left as a further improvement.



## What I'd build next

- **Automatic re-pass on low-confidence blocks.** Close the triage loop: instead of only flagging for the user, the system re-reads flagged frames at higher resolution (or with a targeted prompt) before showing them — a bounded observe -> decide -> act loop. Kept out of the slice because human triage already contains the failure mode, and the re-pass adds a trigger and a merge step.
- **Long-video map-reduce** with section de-duplication, lifting the length cap.
- **Multi-video / series notes** using the source-keyed schema.
- **Durable persistence + job queue** — move notes and job state out of memory into a real store and queue/worker pool, so results survive a reload and processing scales past one process.
- **Offline faithfulness eval** — score each claim against its cited frame/transcript span on a golden set, tracked across prompt changes. This automates, with an LLM judge, the kind of hallucination measurement I did by hand in published video-accessibility research.
- **Per-section re-prompt** — regenerate a single section with its own instruction, rather than re-running the structure pass over the whole video.


## Running it

_(post-build; updated per step)_

Live demo: https://luma-video-notes.onrender.com

Local (the demo URL serves the committed real note; any other URL falls back to a
hand-authored fixture. The server reads committed artifacts and never calls Anthropic at
request time, so no `ANTHROPIC_API_KEY` is required to run it):
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/make_placeholders.py   # optional: regenerates the committed placeholder frames
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```
`pytest -q` runs 41 tests across the pipeline and API contracts. The fixture's placeholder
frame images live under `static/frames/` and are committed; the script above only regenerates them.