"""Step 3B vision: ONE Claude vision call on ONE committed frame -> ONE VisualBlock.

Reads a frame JPEG already on disk (a committed demo artifact), asks Claude to read
the single salient on-screen artifact off the IMAGE, and returns a structured result
(type / content / confidence) -- or an error object. NEVER crashes the caller.

This is the model's job in the pipeline: interpret a frame the deterministic plumbing
already chose (APPROACH.md). It does NOT pick frames, loop, OCR, or structure notes --
those are separate steps. At most ONE block per frame (the dominant artifact).

Config:
  - ANTHROPIC_API_KEY  -- from the environment, or .env for local runs (load_dotenv).
    Deployment sets real env vars and may have NO .env, so the load is best-effort.
  - ANTHROPIC_MODEL    -- overrideable; defaults to claude-sonnet-4-6 (this project's
    Sonnet-tier default for vision). The current Messages API is used with a base64
    image part -- no deprecated completion API.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# load_dotenv is a LOCAL convenience only. Pass an explicit path so dotenv never does
# its find_dotenv() stack-walk (which raises under `python -c` / stdin). Best-effort:
# in deployment there is no .env and real env vars are already set.
try:  # pragma: no cover - trivial best-effort import/load
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

VALID_TYPES = ("code", "table", "diagram", "text")
VALID_CONFIDENCE = ("low", "med", "high")

# Strict-JSON extraction prompt. We ask for the single dominant artifact, typed, as the
# at-most-one VisualBlock the pipeline contract allows per frame. content is markdown so
# the reader renders it directly (fenced code -> <pre>, pipe table -> <table>, else text).
_PROMPT = """You are extracting the single most salient on-screen artifact from ONE frame \
of a technical talk, for verifiable notes. The frame is attached as an image.

{context_block}Identify the ONE dominant artifact visible on screen and return it as a typed block:
- "code": source code. Put it in a fenced block: ```<lang>\\n...\\n``` . Transcribe the \
text as shown; best-effort on indentation (the source frame is the ground truth).
- "table": a data or comparison table. Render it as a GitHub-flavored markdown table.
- "diagram": a diagram, flow, or architecture sketch. Describe its nodes, labels, and \
connections as concise markdown text.
- "text": slide title/bullets/prose with no code, table, or diagram. Extract as markdown text.

Rules:
- Pick the single best-fitting type for the dominant artifact.
- Extract ONLY what is visibly on screen. Do not invent content that is not shown.
- If the frame is a plain talking-head with no artifact, use type "text" with a short note.
- "confidence" is YOUR self-assessment of extraction accuracy: "high", "med", or "low".

Respond with ONLY a JSON object and nothing else -- no prose, no markdown fences around \
the JSON itself:
{{"type": "code|table|diagram|text", "content": "<extracted artifact as markdown>", "confidence": "low|med|high"}}"""


@dataclass
class VisionResult:
    ok: bool = False
    type: str | None = None            # code | table | diagram | text
    content: str | None = None         # extracted artifact as markdown
    confidence: str | None = None      # low | med | high (self-reported, uncalibrated)
    model: str | None = None
    error: str | None = None
    raw: str | None = None             # raw model text (provenance/debug; no keys/headers)
    input_tokens: int | None = None    # usage from the API call (for cost reporting)
    output_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_prompt(transcript_context: str = "", ocr_hint: str = "", candidate_reason: str = "") -> str:
    """Assemble the prompt, optionally prefixing CONTEXT-ONLY hint blocks.

    All hints are framed as NOISY and subordinate to the image: the transcript is context
    the model extracts *around* (not from), and the OCR text is an upstream best-effort the
    model is told to correct/complete -- never copy verbatim where the frame disagrees.
    """
    parts: list[str] = []

    ctx = (transcript_context or "").strip()
    if ctx:
        parts.append(
            "Nearby transcript, for CONTEXT only (extract from the IMAGE, not this text):\n"
            f'"""\n{ctx}\n"""'
        )

    hint = (ocr_hint or "").strip()
    reason = (candidate_reason or "").strip()
    if hint or reason:
        lines = [
            "Hints from the upstream pipeline for THIS frame -- NOISY, for guidance only. "
            "The IMAGE is the ground truth: transcribe what it actually shows, and correct "
            "or complete these hints. Do NOT copy them verbatim where the image disagrees."
        ]
        if reason:
            lines.append(f"- why this frame was auto-selected: {reason}")
        if hint:
            lines.append(
                "- rough OCR of the frame (may be wrong, truncated, or mis-ordered):\n"
                f'"""\n{hint}\n"""'
            )
        parts.append("\n".join(lines))

    context_block = ("\n\n".join(parts) + "\n\n") if parts else ""
    return _PROMPT.format(context_block=context_block)


def _parse_json(text: str) -> dict:
    """Parse the model's reply into a dict. Strips a wrapping code fence if present,
    then falls back to the first {...} span. Raises ValueError if no object is found."""
    t = (text or "").strip()
    fenced = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```$", t, re.DOTALL)
    if fenced:
        t = fenced.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        return json.loads(t[i : j + 1])
    raise ValueError("no JSON object found in model output")


def extract_block(
    image_path: str | os.PathLike,
    transcript_context: str = "",
    ocr_hint: str = "",
    candidate_reason: str = "",
    model: str | None = None,
) -> VisionResult:
    """Run ONE Claude vision call on `image_path` and return a VisionResult.

    `transcript_context`, `ocr_hint`, and `candidate_reason` are optional CONTEXT-ONLY
    hints (see `_build_prompt`): the image is always the ground truth and the model is told
    to correct/complete them, never copy verbatim.

    NEVER raises. On any failure (missing key, missing file, SDK not installed, API
    error, unparseable/invalid output) `ok` is False and `error` carries the cause.
    """
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    res = VisionResult(model=model)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        res.error = "ANTHROPIC_API_KEY not set (looked in environment and .env)"
        return res

    path = Path(image_path)
    if not path.is_file():
        res.error = f"image not found: {path}"
        return res

    try:
        import anthropic
    except ImportError as e:
        res.error = f"anthropic SDK not installed ({e}); add it to requirements.txt"
        return res

    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    try:
        image_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": _build_prompt(transcript_context, ocr_hint, candidate_reason)},
                    ],
                }
            ],
        )
    except Exception as e:  # typed anthropic.* errors all subclass Exception; surface the cause
        res.error = f"{type(e).__name__}: {getattr(e, 'message', e)}"
        return res

    usage = getattr(message, "usage", None)
    res.input_tokens = getattr(usage, "input_tokens", None)
    res.output_tokens = getattr(usage, "output_tokens", None)

    text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()
    res.raw = text
    try:
        parsed = _parse_json(text)
    except Exception as e:
        res.error = f"could not parse model output as JSON: {e}"
        return res

    typ = str(parsed.get("type", "")).strip().lower()
    conf = str(parsed.get("confidence", "")).strip().lower()
    if conf == "medium":
        conf = "med"
    content = parsed.get("content")

    if typ not in VALID_TYPES:
        res.error = f"model returned invalid type: {typ!r}"
        return res
    if not isinstance(content, str) or not content.strip():
        res.error = "model returned empty content"
        return res
    if conf not in VALID_CONFIDENCE:
        conf = "low"  # unknown/odd -> treat as low confidence (triage: verify this one)

    res.type, res.content, res.confidence, res.ok = typ, content, conf, True
    return res
