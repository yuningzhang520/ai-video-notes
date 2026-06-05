"""Step 6 pure-helper tests (no images, no OCR engine, no network).

Covers the change-aware selection primitives added to app.frame_select:
  - crop_to_slide_region (panel-excluding crop box),
  - segment (cropped-pHash jump -> segment ids),
  - choose_representative (OCR-superset primary + low-text last-frame fallback, ties -> later),
  - collapse_adjacent_reps (fold partial reveal into fuller settled twin).
"""

from PIL import Image

from app.frame_select import (
    SEGMENT_PHASH_THRESHOLD,
    choose_representative,
    collapse_adjacent_reps,
    crop_to_slide_region,
    segment,
)


def test_crop_box_drops_right_quarter():
    cropped = crop_to_slide_region(Image.new("RGB", (640, 360)))
    assert cropped.size == (480, 360)            # left 75% of 640 -> 480px wide, full height
    assert crop_to_slide_region(Image.new("RGB", (1280, 720))).size == (960, 720)


def test_segment_starts_a_new_segment_only_on_a_jump_over_threshold():
    # consecutive distances: 1, 1, 8, 0, 10. Use an EXPLICIT threshold of 7 so the test pins the
    # split LOGIC, not the production default (SEGMENT_PHASH_THRESHOLD is data-tuned and may move).
    phs = ["0000000000000000", "0000000000000001", "0000000000000003",
           "00000000000003ff", "00000000000003ff", "0000000000000000"]
    assert segment(phs, 7) == [0, 0, 0, 1, 1, 2]      # 8>7 and 10>7 split; 1,1,0 do not
    # everything collapses at a huge threshold; everything splits at 0.
    assert segment(phs, 64) == [0, 0, 0, 0, 0, 0]
    assert segment(phs, 0)[:3] == [0, 1, 2]
    # the production default keeps the 8-distance pair together (12 >= 8), only 10 < 12 stays too.
    assert segment(phs, SEGMENT_PHASH_THRESHOLD) == [0, 0, 0, 0, 0, 0]


def test_choose_representative_ocr_superset_picks_the_fuller_frame():
    frames = [
        {"id": "a", "ocrText": "Title\nPros\nCons\nSimple"},
        {"id": "b", "ocrText": "Title\nPros\nCons\nSimple\nHarder\nSingle point of failure"},
    ]
    rep, reason = choose_representative(frames)
    assert rep["id"] == "b" and "OCR-superset" in reason


def test_choose_representative_tie_breaks_to_later():
    frames = [{"id": "x", "ocrText": "aa\nbb"}, {"id": "y", "ocrText": "cc\ndd"}]
    rep, _ = choose_representative(frames)
    assert rep["id"] == "y"                       # equal completeness -> later wins


def test_choose_representative_low_text_falls_back_to_last_frame():
    frames = [{"id": "p", "ocrText": ""}, {"id": "q", "ocrText": "single"}, {"id": "r", "ocrText": ""}]
    rep, reason = choose_representative(frames)
    assert rep["id"] == "r" and "last settled frame" in reason


def test_choose_representative_single_frame():
    rep, reason = choose_representative([{"id": "z", "ocrText": "whatever"}])
    assert rep["id"] == "z" and "sole frame" in reason


def test_collapse_folds_partial_reveal_into_fuller_settled():
    reps = [
        {"id": "r1", "ocrText": "Single agent\nPros\nCons\nSimple\nDebug\nLatency\nBig prompt"},
        {"id": "r2", "ocrText": "Single agent\nPros\nCons\nSimple\nDebug\nLatency\nBig prompt\nHarder\nSPOF"},
    ]
    kept, drops = collapse_adjacent_reps(reps)
    assert [k["id"] for k in kept] == ["r2"]                       # fuller kept
    assert drops == [{"dropped": "r1", "kept": "r2", "overlap": 1.0}]


def test_collapse_keeps_distinct_adjacent_reps():
    reps = [
        {"id": "a", "ocrText": "Single agent\nPros\nCons\nSimple"},
        {"id": "b", "ocrText": "Sequential agent\nWorkflow\nAssembly line\nOrder"},
    ]
    kept, drops = collapse_adjacent_reps(reps)
    assert {k["id"] for k in kept} == {"a", "b"} and drops == []


def test_collapse_chains_a_multistage_reveal_to_one():
    reps = [
        {"id": "s1", "ocrText": "Wrap up\nSingle"},
        {"id": "s2", "ocrText": "Wrap up\nSingle\nSequential"},
        {"id": "s3", "ocrText": "Wrap up\nSingle\nSequential\nParallel"},
    ]
    kept, drops = collapse_adjacent_reps(reps)
    assert [k["id"] for k in kept] == ["s3"] and len(drops) == 2   # whole reveal -> fullest


def test_collapse_tie_keeps_the_earlier_rep():
    reps = [{"id": "e1", "ocrText": "aa\nbb\ncc"}, {"id": "e2", "ocrText": "aa\nbb\ncc"}]
    kept, drops = collapse_adjacent_reps(reps)
    assert [k["id"] for k in kept] == ["e1"]                       # equal -> keep earlier, drop later
    assert drops == [{"dropped": "e2", "kept": "e1", "overlap": 1.0}]
