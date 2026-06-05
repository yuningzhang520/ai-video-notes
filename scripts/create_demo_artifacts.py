"""Generate the committed DEMO ARTIFACT PACK via CHANGE-AWARE frame selection (Step 6).

Why this exists: Step 3A's host probe proved request-time YouTube fetch does NOT survive
deployment (cloud IPs are bot-walled), while a residential IP works -- so we pre-extract REAL
frames + transcript locally and commit them, and the deployed server reads them reproducibly.

Step 6 replaces the old fixed-interval sampler (which caught slides mid-reveal) with a
deterministic, content-change-aware selector, ground-truthed in work/probe_sampling:

    dense 0.5 fps sample  ->  segment on CROPPED-pHash jumps  ->  one representative per segment
                          ->  fold adjacent OCR-superset duplicates  ->  commit the representatives

Whole-frame pHash is unusable here (a presenter-video panel on every slide never stops moving),
so the SEGMENTATION distance is computed on the slide region only (left 75%) -- a crop that
conditions the distance signal ONLY; the committed frame, OCR, vision, and verify use the full
frame. Threshold 12 keeps a line-by-line reveal inside one segment so within-segment selection
picks the fully-revealed frame (the single-agent Pros/Cons fix). See app/frame_select.py.

This runs LOCALLY (residential IP) and writes:
    static/demo/<id>/frames/frame-####.jpg   (the SELECTED representatives; served + shipped)
    data/demo/<id>/manifest.json             (rep id -> timestamp -> served URL + selection provenance)
    data/demo/<id>/contact_sheet.jpg         (visual review of the representatives)
It reuses the committed transcript.json (same video) and the gitignored downloaded video / dense
frames if already present (resume). It resets ONLY static/demo/<id>/frames -- never other data.

Run from the repo root:  python scripts/create_demo_artifacts.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.frame_select import (  # noqa: E402
    ADJACENT_OVERLAP,
    MIN_TEXT_SCORE,
    SEGMENT_PHASH_THRESHOLD,
    SLIDE_REGION_FRAC,
    choose_representative,
    collapse_adjacent_reps,
    compute_phash,
    ocr_signals,
    segment,
    segmentation_phash,
)
from app.ingest import parse_video_id  # noqa: E402

VIDEO_ID = "GDm_uH6VxPY"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

# --- Dense sampling knobs (the base sample the selector runs over) -----------
SAMPLE_FPS = 0.5             # one frame every 2.0s
STEP_SEC = 1.0 / SAMPLE_FPS  # 2.0s
OFFSET_SEC = STEP_SEC / 2    # 1.0s -- centre of each bucket
FRAME_W, FRAME_H = 640, 360
JPEG_QUALITY = 6

DOWNLOAD_TIMEOUT_SEC = 180
FFMPEG_TIMEOUT_SEC = 30
SOCKET_TIMEOUT_SEC = 30
MAX_FILESIZE = "300M"
_FORMAT = "bv*[height<=480]/b[height<=480]/bv*/b/worst"

FRAMES_DIR = REPO_ROOT / "static" / "demo" / VIDEO_ID / "frames"
DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
WORK_DIR = REPO_ROOT / "work" / "demo_artifacts" / VIDEO_ID         # gitignored
DENSE_DIR = WORK_DIR / "dense"                                      # gitignored dense base sample
SIGNALS_CACHE = WORK_DIR / "dense_signals.json"                    # gitignored OCR/phash cache
SERVED_PREFIX = f"/static/demo/{VIDEO_ID}/frames"


def _safe_reset(path: Path) -> None:
    """Remove + recreate `path`, but ONLY a demo-artifact dir under the repo carrying the video id
    -- a guard so a typo can never wipe static/frames/ or anything unrelated."""
    resolved = path.resolve()
    assert REPO_ROOT in resolved.parents, f"refusing to reset outside repo: {resolved}"
    assert VIDEO_ID in resolved.parts, f"refusing to reset a non-demo dir: {resolved}"
    assert resolved != (REPO_ROOT / "static" / "frames").resolve(), "refusing to touch static/frames"
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def find_video() -> Path:
    """Return a downloaded video for the demo id, downloading only if none is present (resume)."""
    hits = [p for p in WORK_DIR.glob(f"{VIDEO_ID}.*") if p.suffix not in {".part", ".ytdl"} and p.is_file()]
    if hits:
        print(f"[download] reusing {hits[0].relative_to(REPO_ROOT)} (already present)")
        return hits[0]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress", "--no-cache-dir",
        "--no-mtime", "--retries", "3", "--fragment-retries", "3",
        "--socket-timeout", str(SOCKET_TIMEOUT_SEC), "--max-filesize", MAX_FILESIZE,
        "-f", _FORMAT, "-o", str(WORK_DIR / "%(id)s.%(ext)s"),
        "--no-simulate", "--print", "after_move:filepath", VIDEO_URL,
    ]
    print(f"[download] yt-dlp {VIDEO_URL} ...")
    proc = _run(cmd, DOWNLOAD_TIMEOUT_SEC)
    for line in reversed((proc.stdout or "").splitlines()):
        if line.strip() and Path(line.strip()).is_file():
            return Path(line.strip())
    hits = [p for p in WORK_DIR.glob(f"{VIDEO_ID}.*") if p.suffix not in {".part", ".ytdl"}]
    if hits:
        return hits[0]
    raise RuntimeError(f"download failed (exit {proc.returncode}).\n{(proc.stderr or '')[-1500:]}")


def probe_duration(video_path: Path) -> float:
    proc = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)], 30)
    return float((proc.stdout or "0").strip())


def extract_frame(video_path: Path, ts: float, out_path: Path) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-ss", f"{ts}", "-i", str(video_path), "-frames:v", "1",
           "-vf", f"scale={FRAME_W}:{FRAME_H}", "-q:v", str(JPEG_QUALITY), str(out_path)]
    proc = _run(cmd, FFMPEG_TIMEOUT_SEC)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg failed at t={ts}: {(proc.stderr or '').strip()[:500]}")


def dense_sample(video_path: Path, duration: float) -> list[dict]:
    """Extract the dense 0.5 fps base sample to DENSE_DIR (skip frames already there). Returns
    [{id, ts, path}] ordered by time."""
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    out, t, i = [], OFFSET_SEC, 1
    while t < duration:
        fp = DENSE_DIR / f"d-{i:04d}.jpg"
        if not fp.is_file():
            extract_frame(video_path, round(t, 2), fp)
        out.append({"id": f"d-{i:04d}", "ts": round(t, 2), "path": fp})
        t += STEP_SEC
        i += 1
    return out


def dense_signals(dense: list[dict]) -> list[dict]:
    """phash + cropped seg-phash + OCR for every dense frame (cached to SIGNALS_CACHE)."""
    if SIGNALS_CACHE.is_file():
        cached = json.loads(SIGNALS_CACHE.read_text())
        if len(cached) == len(dense):
            print(f"[signals] reusing {SIGNALS_CACHE.relative_to(REPO_ROOT)} ({len(cached)} frames)")
            for c, d in zip(cached, dense):
                c["path"] = str(d["path"])
            return cached
    recs = []
    for k, d in enumerate(dense, 1):
        img = Image.open(d["path"])
        ocr_text, wc, tl, score, conf = ocr_signals(img)
        recs.append({"id": d["id"], "ts": d["ts"], "path": str(d["path"]),
                     "phash": compute_phash(img), "segPhash": segmentation_phash(img),
                     "ocrText": ocr_text, "wordCount": wc, "textLength": tl,
                     "textScore": score, "ocrConfidence": conf})
        if k % 40 == 0:
            print(f"[signals] {k}/{len(dense)} OCR'd")
    SIGNALS_CACHE.write_text(json.dumps([{k: v for k, v in r.items() if k != "path"} for r in recs],
                                        indent=2) + "\n")
    return recs


def select_representatives(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Segment (cropped-pHash, T) -> one representative per segment -> fold adjacent OCR-superset
    duplicates. Returns (kept_reps_in_time_order, collapse_drops)."""
    recs = sorted(recs, key=lambda r: r["ts"])
    seg_ids = segment([r["segPhash"] for r in recs], SEGMENT_PHASH_THRESHOLD)
    for r, s in zip(recs, seg_ids):
        r["segmentId"] = s
    reps = []
    for sid in range(max(seg_ids) + 1):
        members = [r for r in recs if r["segmentId"] == sid]
        rep, reason = choose_representative(members)
        rep = dict(rep)
        rep["selectionReason"] = reason
        rep["segMemberTimestamps"] = [m["ts"] for m in members]
        reps.append(rep)
    kept, drops = collapse_adjacent_reps(reps, ADJACENT_OVERLAP)
    kept.sort(key=lambda r: r["ts"])
    return kept, drops


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def make_contact_sheet(frames: list[dict], out_path: Path, cols: int = 6) -> None:
    thumb_w, thumb_h, label_h, gap = 213, 120, 16, 4
    cell_w, cell_h = thumb_w, thumb_h + label_h
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap),
                      ImageColor.getrgb("#0e1220"))
    draw = ImageDraw.Draw(sheet)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        x, y = gap + c * (cell_w + gap), gap + r * (cell_h + gap)
        sheet.paste(Image.open(REPO_ROOT / fr["_disk"]).resize((thumb_w, thumb_h)), (x, y))
        draw.text((x + 3, y + thumb_h + 2), f'{fr["id"]}  {mmss(fr["timestampSec"])}', fill="#d6deec")
    sheet.save(out_path, "JPEG", quality=80)


def main() -> None:
    if parse_video_id(VIDEO_URL) != VIDEO_ID:
        raise SystemExit(f"video id mismatch for {VIDEO_URL}")

    transcript_path = DATA_DIR / "transcript.json"
    if not transcript_path.is_file():
        raise SystemExit(f"missing committed transcript {transcript_path}; fetch it first (residential IP)")
    seg_count = json.loads(transcript_path.read_text()).get("segmentCount", 0)

    video_path = find_video()
    duration = probe_duration(video_path)
    dense = dense_sample(video_path, duration)
    print(f"[dense] {len(dense)} frames @ {SAMPLE_FPS} fps over {mmss(duration)}")
    recs = dense_signals(dense)
    reps, drops = select_representatives(recs)
    by_id = {r["id"]: r for r in recs}
    candidates = [r for r in reps if r["textScore"] >= MIN_TEXT_SCORE]
    print(f"[select] {len(dense)} dense -> {max(r['segmentId'] for r in recs) + 1} segments "
          f"-> {len(reps)} representatives ({len(drops)} folded) -> {len(candidates)} pass the text gate")

    # Commit ONLY the representatives: copy each dense JPEG, renumbered in time order.
    _safe_reset(FRAMES_DIR)
    frames_meta = []
    for j, r in enumerate(reps, 1):
        fid = f"frame-{j:04d}"
        disk_rel = f"static/demo/{VIDEO_ID}/frames/{fid}.jpg"
        shutil.copyfile(by_id[r["id"]]["path"], REPO_ROOT / disk_rel)
        frames_meta.append({
            "id": fid, "timestampSec": r["ts"], "imagePath": f"{SERVED_PREFIX}/{fid}.jpg",
            "phash": r["phash"], "segmentId": r["segmentId"], "selectionReason": r["selectionReason"],
            "denseSourceId": r["id"], "segMemberTimestamps": r["segMemberTimestamps"],
            "ocrText": r["ocrText"], "wordCount": r["wordCount"], "textLength": r["textLength"],
            "textScore": r["textScore"], "ocrConfidence": r["ocrConfidence"],
            "_disk": disk_rel,
        })
    print(f"[frames] wrote {len(frames_meta)} representative JPEGs -> {FRAMES_DIR.relative_to(REPO_ROOT)}")

    make_contact_sheet(frames_meta, DATA_DIR / "contact_sheet.jpg")

    # Renumber collapse drops to the committed ids (drops carry dense ids).
    dense_to_new = {f["denseSourceId"]: f["id"] for f in frames_meta}
    collapse_drops = [{"dropped": d["dropped"], "kept": dense_to_new.get(d["kept"], d["kept"]),
                       "overlap": d["overlap"]} for d in drops]

    manifest_frames = [{k: v for k, v in f.items() if not k.startswith("_")} for f in frames_meta]
    (DATA_DIR / "manifest.json").write_text(json.dumps({
        "videoId": VIDEO_ID, "videoUrl": VIDEO_URL, "durationSec": round(duration, 2),
        "samplingFps": SAMPLE_FPS, "frameSize": f"{FRAME_W}x{FRAME_H}",
        "selection": {
            "method": "change-aware: dense -> cropped-pHash segment -> OCR-superset representative",
            "denseFrameCount": len(dense), "segThreshold": SEGMENT_PHASH_THRESHOLD,
            "slideRegionFrac": SLIDE_REGION_FRAC, "adjacentOverlap": ADJACENT_OVERLAP,
            "segments": max(r["segmentId"] for r in recs) + 1, "collapseDrops": collapse_drops,
        },
        "frameCount": len(manifest_frames), "transcriptSegmentCount": seg_count,
        "frames": manifest_frames,
    }, indent=2) + "\n")

    print("\n=== representative pack ready ===")
    print(f"  representatives: {len(manifest_frames)}  (text-gate candidates: {len(candidates)})")
    print(f"  size: {sum((REPO_ROOT / f['_disk']).stat().st_size for f in frames_meta) / 1024:.0f} KiB")
    print(f"  wrote: {(DATA_DIR / 'manifest.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
