# Luma Take-Home

> **This submission: Video -> Verifiable Visual Notes.** See [APPROACH.md](APPROACH.md)
> for what it is and why. Build status is tracked in [PLAN.md](PLAN.md).
>
> **Run locally** — the demo URL serves the committed real note; any other URL falls back to a
> hand-authored fixture. The server reads committed artifacts and never calls Anthropic at request
> time, so no `ANTHROPIC_API_KEY` is needed to run it:
> ```bash
> python3 -m venv venv && source venv/bin/activate
> pip install -r requirements.txt
> uvicorn app.main:app --reload --port 8000   # then open http://localhost:8000
> pytest -q                                    # 41 tests: pipeline + API contracts
> ```
> The fixture's placeholder frame images are committed under `static/frames/`;
> `python scripts/make_placeholders.py` regenerates them. `ANTHROPIC_API_KEY` is used only by the
> build-time scripts, never at runtime.
>
> **Run with Docker** — single containerized service, serving the committed real note for the demo URL (fixture for other URLs):
> ```bash
> docker build -t visual-notes .
> docker run --rm -p 8000:8000 visual-notes               # -> http://localhost:8000
> docker run --rm -e PORT=9000 -p 9000:9000 visual-notes  # honors a platform-injected $PORT
> ```
> The container binds `0.0.0.0:$PORT` (default `8000`). Health check: `GET /healthz` -> `{"status":"ok"}`.
> The image ships the committed demo artifacts under `data/`, and carries `ffmpeg` + `nodejs`
> (ingest probe / build-time frame extraction) and `tesseract-ocr` (upstream OCR for artifact
> generation / optional full rebuild — not used at runtime, and not required for the gate rerun).
>
> **Deploy to a container host (Render / Railway / Fly):** push this repo, point the
> platform at the `Dockerfile`, and set the health check path to `/healthz`. The platform
> injects `$PORT`, which the container already honors — no start command override needed.
> Paste the resulting public URL into APPROACH.md.
>
> **Build-time frame analysis — the reviewer-rerunnable gate:**
> ```bash
> python scripts/analyze_demo_frames.py   # -> data/demo/GDm_uH6VxPY/frame_analysis.json
> ```
> Applies the OCR/text gate using the committed OCR signals to pick the vision candidates — **no LLM,
> no vision, no network, no Tesseract binary**: it reads the per-frame signals already in
> `manifest.json` (only the `requirements.txt` Python deps are needed). Frame *selection* (dense sample
> → cropped-pHash segments → representatives) and the OCR itself already ran upstream in
> `scripts/create_demo_artifacts.py` (residential IP, not reviewer-rerunnable). The output is committed
> and read by the app; the deployed service does NOT run OCR at request time. Tesseract is needed only
> for that upstream artifact generation (and the optional full from-frames rebuild) — not for this gate
> rerun, and not at runtime.
>
> **Optional: reproduce the AI build from committed frames**
>
> Reviewers don't need to start from a YouTube URL. Cloud YouTube video/caption fetch was measured
> and **blocked from Render / datacenter IPs** (the bot-wall — see APPROACH.md), so the repo commits
> the frames / transcript / manifest / artifacts the build already ran on. The **default review
> path** is just to run the app: it serves the committed built note, with **no YouTube or Anthropic
> calls at runtime**. The **optional rebuild path** re-runs the model steps from those committed
> artifacts with your own key — set `ANTHROPIC_API_KEY` before using `--force`. Then
> `build_visual_blocks.py --force` re-runs Claude vision on the 36 selected candidate frames (from
> the committed representative frame set), and `build_full_note.py --force` re-runs the one structure
> pass. Fresh LLM output is schema-valid but **not byte-identical** to the committed output. Without
> `--force`, both scripts reuse the committed cached artifacts at **0 API calls**, and
> `build_full_note.py` rebuilds `note_full.json` deterministically from cached `section_plan.json`.
> **Do not run `--force` without a key** — the scripts return/write failure artifacts instead of a
> useful rebuild.
> ```bash
> # Default product-review path: no key, no YouTube, no Anthropic
> uvicorn app.main:app --port 8000
>
> # Optional: rerun model steps from committed artifacts
> export ANTHROPIC_API_KEY=...
> # optional: export ANTHROPIC_MODEL=claude-sonnet-4-6
> python scripts/build_visual_blocks.py --force
> python scripts/build_full_note.py --force
>
> # Zero-call deterministic rebuild from cached artifacts
> python scripts/analyze_demo_frames.py
> python scripts/build_full_note.py
> ```

---

The original take-home brief follows.

---

Modern engineering is about directing leverage — tools, judgment, taste — toward real outcomes. This take-home is designed around that.

Pick a problem. Build something that works. You have ~1 working day.

**You must use AI coding tools** — Claude Code, Cursor, Codex, whatever you prefer. These problems are scoped so that AI is necessary to ship something real in a day. We want to see how you direct the tools: how you plan, how you course-correct, what you accept, and what you push back on.

---

## Choose a Problem

Pick the one that excites you most. Each option has a **hardest part** — a technical wall, a product judgment call, a messy real-world detail. That's the part we're paying attention to. If you find yourself avoiding it, you've picked the wrong project.

### 1. Reverse-Engineer an Undocumented API

Pick a website that doesn't have a public API. Reverse-engineer how it really works — auth, request shape, rate limits, pagination, anti-bot — then build a real product on top.

Two hard parts: cracking the system, *and* picking the right product to build with what you've unlocked. A scraped CSV isn't a product. The data you can pull is the constraint — your taste is what turns it into something someone would actually use.

**Crack the system, then ship something people would actually use.**

### 2. Build the Mini-App You'd Actually Use

Pick a small problem in your own life. Build it as a deployable web app where AI does real work in the core feature — structured output, vision, an agent loop — not as a chat panel bolted on.

Two hard parts: the product call (what to include, what to cut, what makes the thing good enough that you'd open it twice) *and* the AI integration that has to hold up in front of a real user (latency, weird inputs, the times the model is wrong, the failure mode that has to feel okay).

**Build the small thing you'd actually open twice.**

### 3. Rebuild the Hard Part — with AI

Pick a feature from an app you admire that you assume took the team months to get right: real-time sync, search ranking, undo history, gesture handling, a vision pipeline, recommendations. Rebuild it. Then use AI to change the equation — replace heuristics with model calls, generate the data, do at runtime what they had to do offline — so you ship in a day what they shipped in a quarter.

Two hard parts: the technical lever (AI changing *how it works*, not just being the tool that wrote it) *and* the product call (knowing what "better" actually means for the feature you picked, and whether your version delivers).

**Rebuild the part that took them months. Let AI do something they couldn't.**

---

## Tips

The candidates who do best don't start by building — they start by getting sharp on the problem. It's easy to either throw everything at the wall or get heads-down on making something work, and miss the more important question: *what's actually worth solving here, and for whom?*

Slow down before you write a line of code. The thinking you do upfront will shape everything.

---

## What We're Looking For

We want real, working software — not a prototype, not a toy. You'll likely focus on a slice of the problem, but that slice should actually work and be something you'd put in front of a user. Show polish where it matters to you — in the UX, the details, the interactions that feel right. Ship a finished product, not a proof of concept.

We expect the result to be better than what an AI would produce on its own with minimal guidance. The AI writes the code; you own the decisions — what to build, how it should work, what to cut, and what to polish. Specifically, we're paying attention to:

- **How you approach new problems** — how you break down ambiguity, decide what to tackle first, and make good decisions with incomplete information
- **How you use AI tools** — not just that you used them, but how you directed them, where you pushed back, and where your judgment shaped the result
- **The unique perspective you bring** — the product instincts, technical taste, or domain insight that made your solution distinct from what anyone else would have built

---

## What to Deliver

### 1. Working software

Build your solution directly in this repo. It should run. Include setup instructions that work in a fresh Linux container — we will run your code in one during review. If you use Docker, provide a `docker-compose.yml` for one-command setup.

**If your project is deployable, deploy it.** We want to experience what you built, not just read about it. A live URL — whether it's a web app, an API endpoint, or a hosted service — goes a long way. Vercel, Railway, Fly, a VPS, whatever works. Include the URL in your APPROACH.md.

A `.env.example` is included with stub keys for providers we have accounts with (Anthropic, OpenAI, ElevenLabs, Google Cloud, AWS). Copy it to `.env`, use whichever keys your solution needs, and document any others.

### 2. APPROACH.md

- What you built and why you picked this problem
- Key decisions and tradeoffs
- What you intentionally left out
- What breaks first under pressure
- What you'd build next

### 3. Video walkthrough

Record a short video (~5 minutes) showing what you built. Demo the key flows — whether that's a UI walkthrough, a CLI session, or hitting your API — explain your decisions, and highlight anything you're particularly proud of. This is your chance to show us the experience through your eyes.

**Paste your video link (Loom, Google Drive, YouTube, etc.) into `video.md`.**

### 4. AI session history

Your AI session logs (Claude Code, Codex, Cursor) are packaged automatically when you run `./submit.sh`. If you used other AI tools (ChatGPT, etc.), export those conversations and include them in your repo before submitting.

This is a required deliverable. We review your AI interaction to understand how you work — how you plan, iterate, and direct the tools.

---

## Getting Started

```bash
# 1. Extract the challenge archive you downloaded
tar xzf challenge.tar.gz && cd *eng-take-home*

# 2. Create your own private repo and push to it
git init && git add -A && git commit -m "initial"
gh repo create my-take-home --private --source=. --push

# 3. Copy the env file and fill in any keys you need
cp .env.example .env
```

Now build your solution. Commit and push as you go.

---

## Submitting

When you're ready, run the submit script from your repo root:

```bash
./submit.sh
```

This handles everything: packages your AI session history, commits and pushes your latest changes, grants reviewer access, and registers your submission. You'll see a confirmation when it's done.
