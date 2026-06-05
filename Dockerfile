# Container for the app. Serves the API + static reader, and (Step 3A) carries
# ffmpeg so the ingest probe can extract frames on the host. yt-dlp is a Python
# dependency (requirements.txt), not a system package. Tesseract (Step 4A text
# gate) is the OCR engine pytesseract drives.
FROM python:3.11-slim

# Keep Python predictable in a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System packages (single layer; clean apt lists to keep the image small):
#   ffmpeg        -- app/ingest.py uses it for frame extraction (Step 3A)
#   nodejs        -- a JS runtime so yt-dlp can run YouTube player JS (passed via
#                    --js-runtimes node); without it yt-dlp warns "no supported
#                    JavaScript runtime found" and some formats go missing.
#   tesseract-ocr -- the OCR engine pytesseract drives for the Step 4A text gate.
#                    Frame selection (scripts/analyze_demo_frames.py) is build-time and
#                    its output is committed, so the deployed app does NOT need OCR at
#                    request time; tesseract is shipped only so reviewers can rerun the
#                    analysis inside the container (matches APPROACH.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to run (and to run the smoke test). Explicit COPY means
# venv/, .env, tokens, dist/, and local caches can never enter the image.
COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY tests ./tests
# Committed demo artifacts (transcript / manifest / note_full.json) -- the
# deployed host can't fetch YouTube, so the jobs flow reads the real Note from here.
# (Frame JPEGs ship under static/demo via the static COPY above.)
COPY data ./data

# Platforms inject $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# exec-form `sh -c` so ${PORT} is expanded by the shell at runtime (NOT passed as
# the literal string "$PORT"). Binds 0.0.0.0 so the container is reachable.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
