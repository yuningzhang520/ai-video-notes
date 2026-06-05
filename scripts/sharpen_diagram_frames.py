"""Sharpen ONLY the diagram frames by re-extracting them at higher resolution.

Why: diagram blocks render as the actual frame image (a diagram's value is its spatial
structure, not its text description), but the committed frames are 640x360 extracted from a
<=480p download -- vision can read the small node labels, a human can't. Code/table/text are
fine at 640x360 because their rendered content comes from EXTRACTION and the frame is just a
thumbnail; a diagram needs the human to read the image, so resolution matters.

This re-extracts JUST the diagram frames from a 1080p stream at native 1920x1080 and
overwrites those committed JPEGs IN PLACE. Surgical + data-derived:
  - the set is read from visual_blocks.json (blocks where type == "diagram") -- NO hardcoded
    list -- with timestamps from manifest.json;
  - it overwrites ONLY those frame files. It NEVER wipes the frames dir, touches the other
    frames, or writes any JSON. No vision, no structure pass, no 4A re-run.

It changes only image BYTES, so nothing downstream needs updating: every committed JSON
(manifest / frame_analysis / visual_blocks / note_full) references these frames by PATH and
the reader scales the <img> via CSS. Two latent inconsistencies are left as-is BY DESIGN
(no data change, both harmless):
  - manifest.json's global "frameSize" still reads 640x360 -- the diagram frames are now
    1920x1080. It's a cosmetic descriptor the app never reads.
  - frame_analysis.json's phash for these frames was computed from the OLD 640x360 bytes and
    won't match a fresh scripts/analyze_demo_frames.py run. Harmless: 4A is not re-run and
    phash is build-time-only (dedup). Recorded here as an honest note.

IMPORTANT: re-extracting at 1920x1080 only helps if the SOURCE is actually 1080p. The
originals came from a <=480p download -- that is what made the labels unreadable -- so this
downloads a <=1080p stream and extracts at native resolution. If YouTube serves a smaller
stream from your connection, the script WARNS and the output would merely be upscaled (no
sharper labels).

Run LOCALLY on a residential IP (YouTube bot-walls cloud IPs -- same as create_demo_artifacts.py):
    python scripts/sharpen_diagram_frames.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.ingest import parse_video_id  # noqa: E402

VIDEO_ID = "GDm_uH6VxPY"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
FRAMES_DIR = REPO_ROOT / "static" / "demo" / VIDEO_ID / "frames"
MANIFEST_PATH = DATA_DIR / "manifest.json"
BLOCKS_PATH = DATA_DIR / "visual_blocks.json"
WORK_DIR = REPO_ROOT / "work" / "sharpen_diagrams" / VIDEO_ID  # gitignored

# Native output: 1920x1080 from a 1080p source -> maximum label legibility, no upscaling.
# -q:v 3 keeps text edges sharp (lower = better); only a couple of frames, so the bytes are
# trivial even at full res.
OUT_W, OUT_H = 1920, 1080
JPEG_QUALITY = 3

# Best video stream up to 1080p (the originals were <=480p -- THAT is what made labels
# unreadable). Video-only is enough; fall through so format selection never hard-fails.
_FORMAT = "bv*[height<=1080]/b[height<=1080]/bv*/b"

DOWNLOAD_TIMEOUT_SEC = 240   # 1080p of a short talk
FFMPEG_TIMEOUT_SEC = 30      # per single-frame extraction
SOCKET_TIMEOUT_SEC = 30
MAX_FILESIZE = "600M"        # runaway guard only


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _safe_reset(path: Path) -> None:
    """Remove + recreate `path`, but ONLY a work/ dir carrying the video id -- a guard so this
    can never wipe static/demo/<id>/frames/ or anything else."""
    resolved = path.resolve()
    assert REPO_ROOT in resolved.parents, f"refusing to reset outside repo: {resolved}"
    assert "work" in resolved.parts and VIDEO_ID in resolved.parts, f"refusing to reset {resolved}"
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def diagram_targets() -> list[tuple[str, float]]:
    """The frames to sharpen, derived from DATA: visual_blocks.json blocks where type=="diagram",
    with their timestampSec from manifest.json. Returns [(frameId, timestampSec), ...]."""
    blocks = load_json(BLOCKS_PATH).get("blocks", [])
    by_id = {f["id"]: f for f in load_json(MANIFEST_PATH)["frames"]}
    targets = []
    for b in blocks:
        if b.get("type") != "diagram" or b.get("error") is not None:
            continue
        fid = b["frameRef"]
        if fid not in by_id:
            raise SystemExit(f"diagram block {fid} not in manifest -- artifacts out of sync")
        targets.append((fid, float(by_id[fid]["timestampSec"])))
    return targets


def download_video() -> Path:
    """Download the demo video (<=1080p, video-only) into the gitignored work dir."""
    _safe_reset(WORK_DIR)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-progress", "--no-cache-dir", "--no-mtime",
        "--retries", "3", "--fragment-retries", "3",
        "--socket-timeout", str(SOCKET_TIMEOUT_SEC),
        "--max-filesize", MAX_FILESIZE,
        "-f", _FORMAT,
        "-o", str(WORK_DIR / "%(id)s.%(ext)s"),
        "--no-simulate", "--print", "after_move:filepath",
        VIDEO_URL,
    ]
    print(f"[download] yt-dlp (<=1080p) {VIDEO_URL} -> {WORK_DIR.relative_to(REPO_ROOT)} ...")
    proc = _run(cmd, DOWNLOAD_TIMEOUT_SEC)
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line and Path(line).is_file():
            return Path(line)
    hits = [p for p in WORK_DIR.glob(f"{VIDEO_ID}.*") if p.suffix not in {".part", ".ytdl"}]
    if hits:
        return hits[0]
    raise SystemExit(f"download failed (exit {proc.returncode}).\n{(proc.stderr or '')[-1500:]}")


def probe_height(video_path: Path) -> tuple[int | None, int | None]:
    """The downloaded stream's (width, height), so we can confirm it's actually high-res."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path),
    ]
    out = (_run(cmd, 30).stdout or "").strip()
    try:
        w, h = out.split("x")
        return int(w), int(h)
    except Exception:
        return None, None


def extract_frame(video_path: Path, ts: float, out_path: Path) -> None:
    """Extract ONE 1920x1080 JPEG at `ts`, OVERWRITING out_path. `-ss` before `-i` is a fast,
    accurate seek in modern ffmpeg (decodes to the requested timestamp, not just the keyframe)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{ts}", "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={OUT_W}:{OUT_H}",
        "-q:v", str(JPEG_QUALITY),
        str(out_path),
    ]
    proc = _run(cmd, FFMPEG_TIMEOUT_SEC)
    if proc.returncode != 0 or not out_path.is_file():
        raise SystemExit(f"ffmpeg failed at t={ts}: {(proc.stderr or '').strip()[:500]}")


def dims(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def main() -> None:
    if parse_video_id(VIDEO_URL) != VIDEO_ID:
        raise SystemExit(f"video id mismatch for {VIDEO_URL}")
    for p in (MANIFEST_PATH, BLOCKS_PATH):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run Steps 3B-prep / 4B first")

    targets = diagram_targets()
    if not targets:
        raise SystemExit("no diagram blocks in visual_blocks.json -- nothing to sharpen")
    print("[targets] " + str(len(targets)) + " diagram frame(s) (visual_blocks.json type==diagram): "
          + ", ".join(f"{fid}@{mmss(ts)}" for fid, ts in targets))

    # Snapshot the committed (old) frames so we can report before/after.
    before = {}
    for fid, _ in targets:
        f = FRAMES_DIR / f"{fid}.jpg"
        if not f.is_file():
            raise SystemExit(f"missing committed frame {f}")
        before[fid] = (dims(f), f.stat().st_size)

    video = download_video()
    sw, sh = probe_height(video)
    print(f"[download] got {video.name}  source stream = {sw}x{sh}")
    if sh and sh < OUT_H:
        print(f"[warn] source is only {sh}p -- extracting to {OUT_H}p will UPSCALE (labels will NOT "
              "sharpen). Re-run from a connection where a 1080p stream is available before committing.")

    for fid, ts in targets:
        out = FRAMES_DIR / f"{fid}.jpg"
        extract_frame(video, ts, out)  # overwrite IN PLACE
        w, h = dims(out)
        print(f"[extract] {fid} @ {mmss(ts)} -> {out.relative_to(REPO_ROOT)}  {w}x{h}  {out.stat().st_size / 1024:.0f} KiB")

    print("\n=== sharpened (image bytes only -- no JSON, no other frames touched) ===")
    for fid, _ in targets:
        f = FRAMES_DIR / f"{fid}.jpg"
        (bw, bh), bsz = before[fid]
        aw, ah = dims(f)
        print(f"  {fid}: {bw}x{bh} {bsz / 1024:.0f}KiB  ->  {aw}x{ah} {f.stat().st_size / 1024:.0f}KiB")
    print(f"  downloaded video kept in (gitignored): {WORK_DIR.relative_to(REPO_ROOT)}")
    print("  note: manifest.frameSize (640x360) and frame_analysis phash for these frames are now")
    print("        stale vs the new bytes -- harmless (4A not re-run; both are build-time only).")
    print("  verify: `git status --short` should show ONLY the diagram .jpg(s) changed.")


if __name__ == "__main__":
    main()
