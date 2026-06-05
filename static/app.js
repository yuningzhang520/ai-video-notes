"use strict";

// Reader. Flow: POST a job for the demo video, poll until done, GET the Note, render it. The
// YouTube player and the notes are co-resident, so verifying a bullet seeks the player and
// shows its source -- a frame + that moment's transcript (visual), or the transcript span
// (spoken) -- entirely client-side, never round-tripping the server.
//
// Note shape (5A): section = { title, gist, bullets[] }; bullet = { text, anchor, blocks[] }.
// anchor is null (ungrounded) or { kind:"frame"|"transcript", timestampSec, frameRef, transcriptText }.

const DEMO_URL = "https://www.youtube.com/watch?v=GDm_uH6VxPY";
const DEMO_VIDEO_ID = "GDm_uH6VxPY";
const POLL_MS = 400;
const LARGE_BLOCK_CHARS = 400;     // beyond this, a block is collapsed-by-default
const LARGE_BLOCK_LINES = 10;
const SENTENCE_CAP = 220;          // max transcript-quote length before truncating with "…"
const SENTENCE_MIN = 70;           // below this, extend the quote to a second sentence

// ---- YouTube IFrame Player ----------------------------------------------
let player = null;
let playerReady = false;
let pendingSeek = null;

window.onYouTubeIframeAPIReady = function () {
  player = new YT.Player("player", {
    videoId: DEMO_VIDEO_ID,
    playerVars: {
      rel: 0,
      modestbranding: 1,
      enablejsapi: 1,
      origin: window.location.origin,
    },
    events: {
      onReady: function () {
        playerReady = true;
        if (pendingSeek !== null) { seekPlayer(pendingSeek); pendingSeek = null; }
      },
    },
  });
};

(function loadYouTubeApi() {
  const tag = document.createElement("script");
  tag.src = "https://www.youtube.com/iframe_api";
  document.head.appendChild(tag);
})();

function seekPlayer(seconds) {
  if (playerReady && player && player.seekTo) {
    player.seekTo(seconds, true);
    player.playVideo();
  } else {
    pendingSeek = seconds;
  }
}

// ---- Data ----------------------------------------------------------------
let framesById = {};
let transcriptSegments = [];   // [{start, text}] -- the slim served transcript
let transcriptText = "";       // every segment concatenated (a sentence spans caption cuts)
let segOffsets = [];           // [{start, offset}] -- each segment's char offset in transcriptText
let tocScrollHandler = null;   // ToC scroll-spy handler (removed + rebuilt per render)
let tocUnlockHandler = null;   // resumes the spy after the user scrolls (post click-jump)
let tocLocked = false;         // true between a ToC click and the user's next scroll

function byId(id) { return document.getElementById(id); }

async function loadTranscript() {
  try {
    const t = await getJSON("/static/demo/" + DEMO_VIDEO_ID + "/transcript.json");
    transcriptSegments = t.segments || [];
  } catch (e) {
    transcriptSegments = []; // best-effort
  }
  buildTranscriptIndex();
}

// Flatten the transcript into one string + per-segment char offsets, so a timestamp can be
// resolved to a character position and we can walk to sentence boundaries (which cross the
// caption-segment cuts that make the raw snippets read mid-sentence).
function buildTranscriptIndex() {
  const parts = [];
  segOffsets = [];
  let offset = 0;
  for (const s of transcriptSegments) {
    const t = (s.text || "").replace(/\s+/g, " ").trim();
    if (!t) continue;
    segOffsets.push({ start: s.start, offset: offset });
    parts.push(t);
    offset += t.length + 1; // + the joining space
  }
  transcriptText = parts.join(" ");
}

function _charOffsetForTime(ts) {
  let best = null;
  let bestD = Infinity;
  for (const e of segOffsets) {
    const d = Math.abs(e.start - ts);
    if (d < bestD) { bestD = d; best = e; }
  }
  return best ? best.offset : 0;
}

// One or two COMPLETE sentences around `ts`: walk back to just after the previous . ? ! and
// forward to the next one (extend to a 2nd sentence if the first is short), cap the length and
// add "…" ONLY if actually cut. Used for BOTH visual ("what was said") and spoken quotes, so
// they always read as whole sentences instead of caption fragments.
function transcriptSentenceAt(ts) {
  if (!transcriptText) return null;
  const pos = Math.min(_charOffsetForTime(ts), transcriptText.length - 1);

  let start = 0;
  for (let i = pos - 1; i >= 0; i--) {
    if (/[.?!]/.test(transcriptText[i])) { start = i + 1; break; }
  }
  while (start < transcriptText.length && /\s/.test(transcriptText[start])) start++;

  let end = transcriptText.length;
  for (let i = Math.max(pos, start); i < transcriptText.length; i++) {
    if (/[.?!]/.test(transcriptText[i])) { end = i + 1; break; }
  }
  if (end - start < SENTENCE_MIN) {              // first sentence is short -> add the next one
    for (let i = end; i < transcriptText.length; i++) {
      if (/[.?!]/.test(transcriptText[i])) { end = i + 1; break; }
    }
  }

  let text = transcriptText.slice(start, end).trim();
  if (!text) return null;
  if (text.length > SENTENCE_CAP) text = text.slice(0, SENTENCE_CAP).replace(/\s+\S*$/, "").trim() + "…";
  return text;
}

// ---- Job lifecycle -------------------------------------------------------
async function run() {
  const statusEl = byId("status");
  try {
    const created = await postJSON("/api/jobs", { url: DEMO_URL });
    const jobId = created.job_id;

    statusEl.textContent = "Processing...";
    let status = "processing";
    while (status === "processing") {
      await sleep(POLL_MS);
      status = (await getJSON("/api/jobs/" + jobId)).status;
    }
    if (status !== "done") throw new Error("job ended with status: " + status);

    const note = await getJSON("/api/jobs/" + jobId + "/result");
    await loadTranscript();
    render(note);
  } catch (err) {
    statusEl.textContent = "Failed to load notes: " + err.message;
  }
}

// ---- Rendering -----------------------------------------------------------
function render(note) {
  byId("video-title").textContent = note.video.title;
  framesById = {};
  (note.frames || []).forEach(function (f) { framesById[f.id] = f; });

  // Honest badge: this build serves real pre-extracted demo artifacts.
  const badge = document.querySelector(".badge-fixture, .badge-demo");
  if (badge) {
    const real = (note.frames || []).some(function (f) { return (f.imagePath || "").indexOf("/static/demo/") === 0; });
    badge.textContent = real ? "demo" : "fixture";
    badge.className = real ? "badge-demo" : "badge-fixture";
    badge.title = real ? "Real pre-extracted frames + transcript (committed demo artifacts)." : "Fixture data.";
  }

  // Reset the source panel.
  const oldFig = byId("frame-fig"); if (oldFig) oldFig.remove();
  const oldContent = byId("frame-panel").querySelector(".source-content"); if (oldContent) oldContent.remove();
  const empty = byId("frame-empty");
  empty.style.display = "";
  empty.textContent =
    "Click a point's ▷ to verify: the player jumps there. A visual point shows its source " +
    "frame and what was said; a spoken point shows the transcript.";

  const root = byId("notes");
  root.innerHTML = "";
  note.sections.forEach(function (sec, i) {
    const el = renderSection(sec);
    el.style.setProperty("--i", i); // staggered page-load reveal
    root.appendChild(el);
  });

  buildToc(note.sections);
}

// ---- Table of contents (built always; CSS only shows it on wide screens >=1100px) --
function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function tocEscHandler(e) {
  if (e.key === "Escape" && document.body.classList.contains("toc-open")) {
    document.body.classList.remove("toc-open");
    const t = document.querySelector(".toc-toggle");
    if (t) t.setAttribute("aria-expanded", "false");
  }
}

// A section-level ToC: a toggle tucked into the header + a left drawer that reflow-pushes the
// content right (CSS enables both only >=1100px). The section currently at the top is
// highlighted by a rAF-throttled scroll-spy. Rebuilt on each render -- tears down the prior one.
function buildToc(sections) {
  if (tocScrollHandler) window.removeEventListener("scroll", tocScrollHandler);
  if (tocUnlockHandler) ["wheel", "touchmove", "keydown"].forEach(function (ev) { window.removeEventListener(ev, tocUnlockHandler); });
  tocScrollHandler = null; tocUnlockHandler = null; tocLocked = false;
  document.querySelectorAll(".toc, .toc-toggle").forEach(function (n) { n.remove(); });
  document.body.classList.remove("toc-open");
  if (!sections || sections.length < 2) return;

  // toggle, tucked into the header's left corner
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "toc-toggle";
  toggle.setAttribute("aria-label", "Toggle table of contents");
  toggle.setAttribute("aria-expanded", "false");
  for (let i = 0; i < 3; i++) {
    const line = document.createElement("span");
    line.className = "ic-line";
    toggle.appendChild(line);
  }
  const topbar = document.querySelector(".topbar");
  if (topbar) topbar.insertBefore(toggle, topbar.firstChild);

  // the drawer
  const nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "Table of contents");
  const heading = document.createElement("div");
  heading.className = "toc-heading";
  heading.textContent = "Contents";
  nav.appendChild(heading);
  const list = document.createElement("div");
  list.className = "toc-list";
  const itemById = {};
  function setActiveToc(id) {
    Object.keys(itemById).forEach(function (k) { itemById[k].classList.toggle("is-active", k === id); });
  }

  sections.forEach(function (sec) {
    if (!sec.id) return;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "toc-item";
    item.textContent = sec.title;
    item.addEventListener("click", function () {
      const target = document.getElementById(sec.id);
      if (!target) return;
      setActiveToc(sec.id); // the clicked section wins -- and stays put (a short trailing section
      tocLocked = true;     // can't scroll to the top) until the USER scrolls (wheel/touch/keys)
      target.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    });
    list.appendChild(item);
    itemById[sec.id] = item;
  });
  nav.appendChild(list);
  document.body.appendChild(nav);

  toggle.addEventListener("click", function () {
    const open = document.body.classList.toggle("toc-open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("keydown", tocEscHandler); // same fn ref -> not stacked across renders

  // scroll-spy: the LAST section whose top has scrolled past the header line -- or the last
  // section at the bottom of the page (short trailing sections can't reach the top). A ToC click
  // locks the highlight to the clicked item until the user scrolls again.
  const headerH = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--header-h"), 10) || 52;
  function updateActiveToc() {
    const line = headerH + 24;
    let activeId = sections[0].id;
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 6) {
      activeId = sections[sections.length - 1].id;
    } else {
      for (const sec of sections) {
        const el = document.getElementById(sec.id);
        if (!el) continue;
        if (el.getBoundingClientRect().top <= line) activeId = sec.id; else break;
      }
    }
    setActiveToc(activeId);
  }
  let ticking = false;
  tocScrollHandler = function () {
    if (tocLocked || ticking) return;
    ticking = true;
    requestAnimationFrame(function () { updateActiveToc(); ticking = false; });
  };
  tocUnlockHandler = function () { tocLocked = false; };
  window.addEventListener("scroll", tocScrollHandler, { passive: true });
  ["wheel", "touchmove", "keydown"].forEach(function (ev) { window.addEventListener(ev, tocUnlockHandler, { passive: true }); });
  updateActiveToc();
}

function renderSection(sec) {
  const el = document.createElement("section");
  el.className = "section";
  if (sec.id) el.id = sec.id;

  const h = document.createElement("h2");
  h.textContent = sec.title;
  el.appendChild(h);

  if (sec.gist) {
    const gist = document.createElement("p");
    gist.className = "gist";
    gist.textContent = sec.gist;
    el.appendChild(gist);
  }

  const list = document.createElement("div");
  list.className = "bullets";
  (sec.bullets || []).forEach(function (b) { list.appendChild(renderBullet(b)); });
  el.appendChild(list);
  return el;
}

function bulletTier(bullet) {
  const a = bullet.anchor;
  if (!a || !bullet.text || !bullet.text.trim() || a.timestampSec === undefined || a.timestampSec === null) return "ungrounded";
  if (a.kind === "frame") return framesById[a.frameRef] ? "visual" : "ungrounded";
  if (a.kind === "transcript") return a.transcriptText && a.transcriptText.trim() ? "spoken" : "ungrounded";
  return "ungrounded";
}

// A bullet group: the point (with its verify chip) + its evidence nested under the same rail.
function renderBullet(bullet) {
  const tier = bulletTier(bullet);
  const el = document.createElement("div");
  el.className = "bullet " + tier;

  const line = document.createElement("div");
  line.className = "bullet-line";

  const txt = document.createElement("span");
  txt.className = "bullet-text";
  txt.textContent = bullet.text;
  line.appendChild(txt);

  if (tier === "ungrounded") {
    const ns = document.createElement("span");
    ns.className = "no-source";
    ns.textContent = "no source";
    ns.title = "No on-screen or spoken source; shown, not click-to-verify.";
    line.appendChild(ns);
  } else {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "verify-chip";
    const ts = bullet.anchor.timestampSec;
    chip.textContent = "▷ " + mmss(ts);
    chip.setAttribute("aria-label",
      (tier === "visual" ? "Verify: show source frame and play from " : "Verify: show transcript and play from ") + mmss(ts));
    chip.addEventListener("click", function () { verifyBullet(bullet); });
    line.appendChild(chip);
  }
  el.appendChild(line);

  (bullet.blocks || []).forEach(function (b) { el.appendChild(renderBlock(b)); });
  return el;
}

function renderBlock(block) {
  const el = document.createElement("div");
  el.className = "block";

  const head = document.createElement("div");
  head.className = "block-head";
  const type = document.createElement("span");
  type.className = "type-badge";
  type.textContent = block.type;
  head.appendChild(type);
  if (block.confidence) {
    const conf = document.createElement("span");
    conf.className = "conf conf-" + block.confidence;
    conf.textContent = block.confidence === "low" ? "low · verify" : "confidence: " + block.confidence;
    head.appendChild(conf);
  }
  el.appendChild(head);

  if (block.type === "diagram") {
    el.appendChild(diagramFigure(block));
    return el;
  }

  const content = block.content || "";
  if (isLargeBlock(content)) {
    el.classList.add("collapsible");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "disclosure";
    btn.setAttribute("aria-expanded", "false");
    const caret = document.createElement("span");
    caret.className = "caret";
    caret.textContent = "▸";
    btn.appendChild(caret);
    btn.appendChild(document.createTextNode(" " + content.split("\n").length + " lines"));

    const collapse = document.createElement("div");
    collapse.className = "collapse";
    const inner = document.createElement("div");
    inner.className = "collapse-inner";
    const body = document.createElement("div");
    body.className = "block-body";
    body.appendChild(renderMarkdown(content));
    inner.appendChild(body);
    collapse.appendChild(inner);

    btn.addEventListener("click", function () {
      const open = el.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    el.appendChild(btn);
    el.appendChild(collapse);
  } else {
    const body = document.createElement("div");
    body.className = "block-body";
    body.appendChild(renderMarkdown(content));
    el.appendChild(body);
  }
  return el;
}

function isLargeBlock(content) {
  return content.length > LARGE_BLOCK_CHARS || content.split("\n").length > LARGE_BLOCK_LINES;
}

function diagramFigure(block) {
  const frame = framesById[block.frameRef];
  const fig = document.createElement("figure");
  fig.className = "block-diagram";
  if (frame) {
    const img = document.createElement("img");
    img.src = frame.imagePath;
    img.alt = block.content || ("Diagram from " + block.frameRef);
    img.title = block.content || "";
    img.loading = "lazy";
    fig.appendChild(img);
  } else {
    fig.appendChild(renderMarkdown(block.content || ""));
  }
  return fig;
}

// ---- The trust moment: seek + compose the source panel (fluid cross-fade) ----------
function verifyBullet(bullet) {
  const a = bullet.anchor;
  if (!a) return;
  seekPlayer(a.timestampSec);
  swapSource(function (c) {
    if (a.kind === "frame") {
      const blk = (bullet.blocks || []).find(function (b) { return b.frameRef === a.frameRef; });
      c.appendChild(frameNode(a.frameRef, a.timestampSec, blk ? blk.type : null));
      const said = transcriptSentenceAt(a.timestampSec);
      if (said) c.appendChild(quoteNode(said, "what was said", null));
    } else if (a.kind === "transcript") {
      // whole-sentence lookup for both paths; fall back to the baked span if transcript didn't load
      const said = transcriptSentenceAt(a.timestampSec) || a.transcriptText;
      c.appendChild(quoteNode(said, "transcript", a.timestampSec));
    }
  });
}

function frameNode(frameRef, ts, type) {
  const frame = framesById[frameRef];
  const fig = document.createElement("figure");
  fig.className = "source-frame";
  if (frame) {
    const img = document.createElement("img");
    img.src = frame.imagePath;
    img.alt = "Source frame " + frameRef;
    img.loading = "lazy";
    fig.appendChild(img);
  }
  const cap = document.createElement("figcaption");
  cap.className = "source-cap";
  const ref = document.createElement("span");
  ref.className = "ref";
  ref.textContent = frameRef;
  cap.appendChild(ref);
  // Caption shows the frame's EXACT original timestamp (frame.timestampSec), data-derived per frame.
  // The verify SEEK still uses the anchor's verify-lead (seekPlayer in verify()) -- unchanged here.
  const capTs = (frame && typeof frame.timestampSec === "number") ? frame.timestampSec : ts;
  cap.appendChild(document.createTextNode(" · " + mmss(capTs) + (type ? " · " + type + " extracted by vision" : "")));
  fig.appendChild(cap);
  return fig;
}

function quoteNode(text, label, ts) {
  const wrap = document.createElement("div");
  const lab = document.createElement("div");
  lab.className = "source-quote-label";
  lab.textContent = label;
  const q = document.createElement("blockquote");
  q.className = "source-quote";
  q.textContent = text; // escaped via textContent
  wrap.appendChild(lab);
  wrap.appendChild(q);
  if (ts !== null && ts !== undefined) {
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = "transcript · " + mmss(ts);
    wrap.appendChild(meta);
  }
  return wrap;
}

// Fade the current source content out, swap, fade the new in.
function swapSource(build) {
  const panel = byId("frame-panel");
  byId("frame-empty").style.display = "none";

  function mount() {
    const c = document.createElement("div");
    c.className = "source-content";
    build(c);
    panel.appendChild(c);
    requestAnimationFrame(function () { c.classList.add("in"); });
  }

  const old = panel.querySelector(".source-content");
  if (!old) { mount(); return; }

  function finalize() {
    if (!old.isConnected) return; // already handled by the other trigger
    old.remove();
    mount();
  }
  old.classList.remove("in");
  old.classList.add("out");
  old.addEventListener("transitionend", finalize, { once: true });
  setTimeout(finalize, 240); // fallback (reduced motion / no transition)
}

// ---- Minimal, XSS-safe markdown -> DOM (escape-first; never innerHTML on content) --
function renderMarkdown(md) {
  const frag = document.createDocumentFragment();
  const lines = (md || "").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      frag.appendChild(preEl(buf.join("\n")));
      continue;
    }
    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
      frag.appendChild(tableEl(rows));
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const el = document.createElement("p");
      el.className = "md-h";
      appendInline(el, h[2]);
      frag.appendChild(el);
      i++;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = document.createElement("ul");
      ul.className = "md-ul";
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const li = document.createElement("li");
        appendInline(li, lines[i].replace(/^\s*[-*]\s+/, ""));
        ul.appendChild(li);
        i++;
      }
      frag.appendChild(ul);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [];
    while (
      i < lines.length && lines[i].trim() &&
      !/^```/.test(lines[i]) && !/^\s*\|/.test(lines[i]) &&
      !/^#{1,6}\s/.test(lines[i]) && !/^\s*[-*]\s+/.test(lines[i])
    ) { para.push(lines[i]); i++; }
    const p = document.createElement("p");
    p.className = "md-p";
    appendInline(p, para.join(" "));
    frag.appendChild(p);
  }
  return frag;
}

function appendInline(el, text) {
  const re = /\*\*([^*]+)\*\*|`([^`]+)`/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) el.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1] !== undefined) {
      const s = document.createElement("strong");
      s.textContent = m[1];
      el.appendChild(s);
    } else {
      const c = document.createElement("code");
      c.textContent = m[2];
      el.appendChild(c);
    }
    last = re.lastIndex;
  }
  if (last < text.length) el.appendChild(document.createTextNode(text.slice(last)));
}

function preEl(text) {
  const pre = document.createElement("pre");
  pre.textContent = text;
  return pre;
}

function tableEl(lines) {
  const rows = lines
    .map(function (l) { return l.trim(); })
    .filter(function (l) { return l.indexOf("|") === 0; })
    .map(function (l) { return l.replace(/^\|/, "").replace(/\|$/, "").split("|").map(function (c) { return c.trim(); }); });
  const body = rows.filter(function (r) {
    return !r.every(function (c) { return /^:?-{1,}:?$/.test(c) || c === ""; });
  });
  const table = document.createElement("table");
  table.className = "md";
  if (body.length) {
    const thead = document.createElement("thead");
    thead.appendChild(rowEl(body[0], "th"));
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    body.slice(1).forEach(function (r) { tbody.appendChild(rowEl(r, "td")); });
    table.appendChild(tbody);
  }
  return table;
}

function rowEl(cells, tag) {
  const tr = document.createElement("tr");
  cells.forEach(function (c) {
    const cell = document.createElement(tag);
    appendInline(cell, c);
    tr.appendChild(cell);
  });
  return tr;
}

// ---- Small utils ---------------------------------------------------------
function mmss(seconds) {
  const s = Math.round(seconds);
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error("POST " + url + " -> " + res.status);
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("GET " + url + " -> " + res.status);
  return res.json();
}

run();
