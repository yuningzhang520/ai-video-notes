"""Generate labeled placeholder frame images for Step 1A.

These are NOT extracted video frames -- no video is downloaded in Step 1A. Each
image is a clearly-marked PLACEHOLDER stamped with the frame id and its real
timestamp in the demo video, so the click-to-verify trust moment is visible end
to end. Step 3 swaps real extracted frames in at the same paths (same frame ids,
same timestamps), against this exact contract.

Run from the repo root:  python scripts/make_placeholders.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.fixture import build_fixture_note  # noqa: E402

OUT_DIR = REPO_ROOT / "static" / "frames"
SIZE = (640, 360)  # 16:9

# A distinct tint per frame so swapping frames is obviously visible in the demo.
TINTS = ["#1f2a44", "#173a2b", "#3a1f2b"]


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def wrap(draw, text, font, max_width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_image(frame, tint, path):
    img = Image.new("RGB", SIZE, ImageColor.getrgb(tint))
    draw = ImageDraw.Draw(img)
    pad = 28

    # Border so the placeholder reads as a framed image.
    draw.rectangle([6, 6, SIZE[0] - 7, SIZE[1] - 7], outline="#5b6b8c", width=2)

    draw.text((pad, pad), "PLACEHOLDER FRAME", fill="#9fb0d0")
    draw.text((pad, pad + 22), f"{frame.id}   t = {mmss(frame.timestamp_sec)}", fill="#ffffff")

    label = frame.vision_description or frame.ocr_text or ""
    y = pad + 70
    for line in wrap(draw, label, None, SIZE[0] - 2 * pad):
        draw.text((pad, y), line, fill="#d6deec")
        y += 18

    draw.text((pad, SIZE[1] - pad - 12), "not a real extracted frame", fill="#7e8db0")

    img.save(path, "JPEG", quality=85)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_fixture_note().frames
    for i, frame in enumerate(frames):
        make_image(frame, TINTS[i % len(TINTS)], OUT_DIR / f"{frame.id}.jpg")


if __name__ == "__main__":
    main()
