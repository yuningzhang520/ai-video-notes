"""Step 3A ingest PROBE -- standalone, NOT wired into the jobs pipeline.

Purpose: de-risk the single highest-risk unknown before building the real
pipeline -- can the deployed host (Render, a datacenter IP) actually pull a
YouTube video and extract frames? yt-dlp from cloud IPs is often throttled or
bot-walled by YouTube, so we prove (or disprove) the fetch + frame-extract path
in isolation, hit the SAME code from a diagnostic endpoint on the live host, and
choose the right fallback BEFORE any vision/structure work depends on it.

Design notes (deliberately throwaway-grade so it's easy to evolve or delete):
  - yt-dlp and ffmpeg are invoked as bounded SUBPROCESSES (not the yt-dlp Python
    API): a subprocess gives a hard wall-clock timeout and lets us capture the
    real stderr tail -- which is the entire point, since we want to learn *why*
    a download fails (geo / age / bot wall / format / missing binary), not just
    that it did.
  - LOW resolution on purpose: we are proving the path, not the quality.
  - Bounded everywhere: max_frames is clamped to <= 3, the download and ffmpeg
    calls each have timeouts, and a max-filesize cap stops a runaway pull.
  - NEVER crashes the caller: every failure path returns ok=False with the
    underlying cause in `error`.

This module does not touch the Note schema, the fixture, or the jobs API.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- Bounds (so a probe can neither hang nor pull something huge) -------------
DOWNLOAD_TIMEOUT_SEC = 90      # hard kill on the yt-dlp subprocess
FFMPEG_TIMEOUT_SEC = 30        # hard kill on the ffmpeg subprocess
MAX_FILESIZE = "200M"          # yt-dlp --max-filesize guard
SOCKET_TIMEOUT_SEC = 30        # yt-dlp per-socket timeout
HARD_MAX_FRAMES = 3            # never extract more than this, whatever the caller asks
STDERR_TAIL_CHARS = 1500       # how much of the real stderr to surface in `error`

# Where probe artifacts go by default: a gitignored, dockerignored work dir.
# Callers normally pass their own out_dir (a temp dir); this is just the base.
_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = _BASE_DIR / "work" / "ingest"

_YT_ID_RE = re.compile(
    r"(?:v=|/shorts/|/embed/|youtu\.be/)([0-9A-Za-z_-]{11})"
)

# Prefer a small, video-bearing format; fall through to "worst" so format
# selection never hard-fails. No "+audio" merge -- we only need frames.
_FORMAT = "worst[height<=360][vcodec!=none]/worst[vcodec!=none]/worst"


@dataclass
class IngestResult:
    ok: bool = False
    video_id: str | None = None
    downloaded_path: str | None = None     # the "downloaded_path_or_bytes" -- a path on disk
    frame_paths: list[str] = field(default_factory=list)
    error: str | None = None
    # Extra diagnostics (the whole point is to learn what happened):
    stage: str = "init"                    # which step we reached: download | extract | done
    download_bytes: int | None = None
    frame_count: int = 0
    diagnostics: dict = field(default_factory=dict)  # versions + tool availability on this host

    def to_dict(self) -> dict:
        return asdict(self)


def _ytdlp_version() -> str | None:
    try:
        import yt_dlp  # local import: only needed for the version string
        return yt_dlp.version.__version__
    except Exception:  # pragma: no cover -- diagnostics must never crash the probe
        return None


def _js_runtime_paths() -> dict:
    """Which JS runtimes this host has. yt-dlp auto-enables only deno; node must
    be enabled explicitly (see _js_runtime_args), so we report both."""
    return {"node": shutil.which("node"), "deno": shutil.which("deno")}


def _js_runtime_args() -> list[str]:
    """Enable the apt-installed node runtime when present. NOT a bypass attempt --
    just makes the installed runtime usable so yt-dlp can run player JS instead of
    warning that no runtime was found. deno (yt-dlp's default) needs no flag."""
    return ["--js-runtimes", "node"] if shutil.which("node") else []


def runtime_diagnostics() -> dict:
    """Host capability snapshot for the probe JSON: yt-dlp version, ffmpeg
    availability, and JS runtime availability/path. Cheap, never raises."""
    return {
        "yt_dlp_version": _ytdlp_version(),
        "ffmpeg_path": _resolve_bin("ffmpeg", "FFMPEG_BIN"),
        "js_runtimes": _js_runtime_paths(),
    }


def parse_video_id(url: str) -> str | None:
    """Best-effort YouTube id from a URL. Returns None if nothing 11-char matches."""
    m = _YT_ID_RE.search(url or "")
    return m.group(1) if m else None


def _tail(*chunks: str | None) -> str:
    """Join captured stdout/stderr and keep only the meaningful tail."""
    text = "\n".join(c.strip() for c in chunks if c and c.strip())
    text = text.strip()
    if len(text) > STDERR_TAIL_CHARS:
        text = "...(truncated)...\n" + text[-STDERR_TAIL_CHARS:]
    return text or "(no stderr captured)"


def _resolve_bin(name: str, env_var: str) -> str | None:
    """Find an executable: explicit env override first (e.g. FFMPEG_BIN), then PATH."""
    override = os.environ.get(env_var)
    if override:
        return override if (Path(override).exists() or shutil.which(override)) else None
    return shutil.which(name)


def fetch_and_sample(
    url: str,
    out_dir: str | os.PathLike,
    every_sec: int = 60,
    max_frames: int = 3,
) -> IngestResult:
    """Download `url` at low resolution with yt-dlp and extract up to `max_frames`
    frames (one every `every_sec` seconds) with ffmpeg, into `out_dir`.

    Returns an IngestResult; NEVER raises. On any failure (download blocked,
    geo/age/bot wall, ffmpeg missing, timeout) `ok` is False and `error` carries
    the real underlying cause (the truncated tail of the tool's stderr).
    """
    max_frames = max(1, min(int(max_frames), HARD_MAX_FRAMES))
    every_sec = max(1, int(every_sec))
    video_id = parse_video_id(url)
    res = IngestResult(video_id=video_id, diagnostics=runtime_diagnostics())

    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        res.error = f"could not create out_dir {out}: {e}"
        return res

    # --- 1) Download (yt-dlp as a bounded subprocess) -------------------------
    res.stage = "download"
    yt_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--no-cache-dir",
        "--no-mtime",
        "--retries", "1",
        "--fragment-retries", "1",
        "--socket-timeout", str(SOCKET_TIMEOUT_SEC),
        "--max-filesize", MAX_FILESIZE,
        # ONE credential-free long-shot at the datacenter-IP bot wall: try the
        # non-web player clients, which sometimes dodge the "confirm you're not a
        # bot" interstitial. Single attempt -- NO cookies, proxies, or retries.
        "--extractor-args", "youtube:player_client=android,ios,tv",
        # Enable the apt-installed node runtime when present (deno is the default).
        *_js_runtime_args(),
        "-f", _FORMAT,
        "-o", str(out / "%(id)s.%(ext)s"),
        # Download AND print the final path (last stdout line) so we don't have to guess it.
        "--no-simulate", "--print", "after_move:filepath",
        url,
    ]
    try:
        proc = subprocess.run(
            yt_cmd,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        res.error = (
            f"yt-dlp timed out after {DOWNLOAD_TIMEOUT_SEC}s "
            f"(host likely throttled/blocked).\n{_tail(_decode(e.stdout), _decode(e.stderr))}"
        )
        return res
    except FileNotFoundError as e:
        res.error = f"could not launch yt-dlp ({sys.executable} -m yt_dlp): {e}"
        return res

    downloaded = _find_downloaded_file(proc.stdout, out, video_id)
    if proc.returncode != 0 or downloaded is None:
        res.error = (
            f"yt-dlp download failed (exit {proc.returncode}).\n"
            f"{_tail(proc.stdout, proc.stderr)}"
        )
        return res

    res.downloaded_path = str(downloaded)
    try:
        res.download_bytes = downloaded.stat().st_size
    except OSError:
        pass

    # --- 2) Extract frames (ffmpeg as a bounded subprocess) -------------------
    res.stage = "extract"
    ffmpeg = _resolve_bin("ffmpeg", "FFMPEG_BIN")
    if ffmpeg is None:
        res.error = (
            "ffmpeg not found (looked at $FFMPEG_BIN then PATH). "
            "Download succeeded but frames cannot be extracted on this host."
        )
        return res

    frame_tmpl = str(out / "frame-%03d.jpg")
    ff_cmd = [
        ffmpeg,
        "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(downloaded),
        "-vf", f"fps=1/{every_sec}",
        "-frames:v", str(max_frames),
        "-q:v", "5",
        frame_tmpl,
    ]
    try:
        ff = subprocess.run(
            ff_cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        res.error = (
            f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC}s.\n"
            f"{_tail(_decode(e.stdout), _decode(e.stderr))}"
        )
        return res

    frames = sorted(str(p) for p in out.glob("frame-*.jpg"))
    if ff.returncode != 0 or not frames:
        res.error = (
            f"ffmpeg frame extraction failed (exit {ff.returncode}, "
            f"{len(frames)} frames).\n{_tail(ff.stdout, ff.stderr)}"
        )
        return res

    # --- Success -------------------------------------------------------------
    res.frame_paths = frames
    res.frame_count = len(frames)
    res.stage = "done"
    res.ok = True
    return res


def _decode(maybe_bytes) -> str | None:
    """TimeoutExpired may carry bytes even when text=True; normalize to str."""
    if maybe_bytes is None:
        return None
    if isinstance(maybe_bytes, bytes):
        return maybe_bytes.decode("utf-8", "replace")
    return str(maybe_bytes)


def _find_downloaded_file(stdout: str | None, out: Path, video_id: str | None) -> Path | None:
    """Resolve the downloaded media file.

    Primary: the last stdout line from `--print after_move:filepath` that points
    at a real file. Fallbacks: glob by parsed video id, then any non-frame,
    non-partial media file in the dir (newest wins).
    """
    if stdout:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line and os.path.isfile(line):
                return Path(line)

    if video_id:
        hits = [p for p in out.glob(f"{video_id}.*") if p.suffix not in {".part", ".ytdl"}]
        if hits:
            return hits[0]

    candidates = [
        p for p in out.iterdir()
        if p.is_file()
        and not p.name.startswith("frame-")
        and p.suffix not in {".part", ".ytdl"}
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None
