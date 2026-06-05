"""Step 3A transcript PROBE -- standalone, NOT wired into the jobs pipeline.

Companion to app/ingest.py. The Render probe showed server-side YouTube *video*
download is bot-walled from the datacenter IP, so the shippable slice falls back
to transcript + pre-extracted committed frames. This module checks the other
half of that fallback: can we pull the demo video's captions from the host?

Captions are fetched over HTTP from YouTube too, so the SAME datacenter-IP block
can hit here (the library raises RequestBlocked / IpBlocked) -- which is exactly
why this is a probe we run on Render, not an assumption. Bounded via a custom
requests session timeout; NEVER raises -- every failure returns ok=False with the
real cause (including the exception type, so a bot wall is identifiable).

Not wired into POST /api/jobs; the real result still comes from the fixture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import requests
from youtube_transcript_api import YouTubeTranscriptApi

# Per-HTTP-request timeout so a probe can't hang on the network.
REQUEST_TIMEOUT_SEC = 20


class _TimeoutSession(requests.Session):
    """A requests Session that injects a default timeout into every call --
    youtube-transcript-api takes an http_client but exposes no timeout knob."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


@dataclass
class TranscriptResult:
    ok: bool = False
    video_id: str | None = None
    segments: list[dict] = field(default_factory=list)  # {text, start, duration}
    text: str | None = None
    segment_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> TranscriptResult:
    """Fetch `video_id`'s transcript via youtube-transcript-api.

    Returns a TranscriptResult; NEVER raises. On failure (no captions, captions
    disabled, IP/bot block, network timeout) `ok` is False and `error` carries the
    exception type + message so the cause is identifiable from the probe JSON.
    """
    res = TranscriptResult(video_id=video_id)
    try:
        api = YouTubeTranscriptApi(http_client=_TimeoutSession(REQUEST_TIMEOUT_SEC))
        fetched = api.fetch(video_id, languages=list(languages))
        segments = [
            {"text": s.text, "start": round(s.start, 2), "duration": round(s.duration, 2)}
            for s in fetched
        ]
        res.segments = segments
        res.segment_count = len(segments)
        res.text = " ".join(s["text"] for s in segments).strip()
        res.ok = bool(segments)
        if not res.ok:
            res.error = "transcript fetched but contained no segments"
    except Exception as e:  # never crash the caller; surface the real cause
        res.error = f"{type(e).__name__}: {e}"
    return res
