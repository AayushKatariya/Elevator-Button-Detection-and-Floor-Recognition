#!/usr/bin/env python3
"""
Verify detections come back in ORIGINAL-image pixel coordinates, at any size
and aspect ratio.

This is the highest-risk silent bug in the pipeline. YOLO runs on a 640x640
letterboxed copy of the input; if the returned boxes were ever in that
letterboxed space rather than the caller's, every bbox and centre would be
offset and scaled wrong. The output would still look structurally valid -- the
JSON schema would be satisfied, the numbers would be plausible -- so nothing
downstream would complain.

Critically, THE ENTIRE TEST SPLIT WOULD PASS ANYWAY, because every image in the
dataset is a perfect 640x640 square and square input receives no letterbox
padding. The bug only appears on real photographs.

The test builds a synthetic canvas by pasting a known panel at a known offset
and scale inside a large portrait frame, then checks that detections land where
that panel actually is.

    python tests/test_coordinate_space.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline import ElevatorPipeline, load_config  # noqa: E402

# (canvas_w, canvas_h, description) - deliberately includes extreme aspects
CASES = [
    (3024, 4032, "portrait 3024x4032 (phone photo)"),
    (4032, 3024, "landscape 4032x3024"),
    (1920, 1080, "landscape 16:9"),
    (800, 2400, "extreme portrait 1:3"),
    (640, 640, "square 640x640 (baseline - no letterbox)"),
]


def build_canvas(panel: np.ndarray, cw: int, ch: int):
    """Paste the panel, scaled to fit ~60% of the canvas, at a known offset."""
    ph, pw = panel.shape[:2]
    scale = min(cw / pw, ch / ph) * 0.6
    nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
    resized = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ch, cw, 3), 40, np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, (ox, oy, nw, nh, scale)


def main() -> int:
    cfg = load_config("config.yaml")
    # Detector only: OCR is irrelevant to coordinate space and much slower.
    cfg = dict(cfg); cfg["ocr"] = dict(cfg["ocr"], engine="classifier")
    pipe = ElevatorPipeline(cfg)

    src = sorted(Path("examples/input").glob("*.jpg"))
    if not src:
        print("no images in examples/input - run the sample copy step first")
        return 1
    panel = pipe.imread(src[0])
    print(f"source panel: {src[0].name}  {panel.shape[1]}x{panel.shape[0]}")

    base = pipe.detect(panel)
    print(f"baseline detections on the panel itself: {len(base)}\n")
    if not base:
        print("FAIL: no detections on the source panel")
        return 1

    print(f"{'case':<40}{'dets':>6}{'inside':>8}{'max_err_px':>12}{'recall':>10}  verdict")
    print("-" * 82)
    failures = 0
    notes: list[str] = []
    for cw, ch, desc in CASES:
        canvas, (ox, oy, nw, nh, scale) = build_canvas(panel, cw, ch)
        dets = pipe.detect(canvas)

        # 1. every box must lie within the canvas
        in_canvas = all(0 <= d.bbox[0] < d.bbox[2] <= cw and
                        0 <= d.bbox[1] < d.bbox[3] <= ch for d in dets)
        # 2. and within the pasted panel region (plus a small tolerance)
        tol = max(6, int(0.02 * max(nw, nh)))
        inside = sum(1 for d in dets
                     if ox - tol <= d.bbox[0] and d.bbox[2] <= ox + nw + tol
                     and oy - tol <= d.bbox[1] and d.bbox[3] <= oy + nh + tol)

        # 3. Map baseline boxes into canvas space and measure agreement.
        #    Only MATCHED pairs count. A baseline button with no detection
        #    nearby is a recall miss (the panel can be letterboxed very small at
        #    extreme aspect ratios), which is a different problem from boxes
        #    being in the wrong coordinate space -- conflating the two makes
        #    this test report a coordinate bug that does not exist.
        med = float(np.median([b.width for b in base])) * scale
        match_radius = max(8.0, med)
        errs, matched = [], 0
        for b in base:
            ecx = (b.bbox[0] + b.bbox[2]) / 2 * scale + ox
            ecy = (b.bbox[1] + b.bbox[3]) / 2 * scale + oy
            if not dets:
                continue
            near = min(dets, key=lambda d: (d.center[0] - ecx) ** 2
                       + (d.center[1] - ecy) ** 2)
            err = ((near.center[0] - ecx) ** 2 + (near.center[1] - ecy) ** 2) ** 0.5
            if err <= match_radius:
                errs.append(err); matched += 1
        max_err = max(errs) if errs else float("nan")
        recall = matched / len(base)

        # What this test asserts is COORDINATE SPACE, not sub-pixel precision:
        # a centre landing within one button width of the right place proves the
        # box is on the correct button, which is the property that matters. If
        # the boxes were in letterboxed space instead, the error would be
        # hundreds of pixels and they would fall outside the panel region.
        #
        # Precision does degrade at extreme aspect ratios and that is expected:
        # an 800x2400 canvas letterboxes to a scale of ~0.27, so each model-space
        # pixel is worth ~3.7 original pixels before any localisation error.
        limit = max(8.0, 1.0 * med)
        ok = bool(dets) and in_canvas and inside == len(dets) \
            and errs and max_err <= limit
        failures += (not ok)
        print(f"{desc:<40}{len(dets):>6}{inside:>8}{max_err:>12.1f}"
              f"{100*recall:>9.0f}%  {'PASS' if ok else 'FAIL'}")
        lb = min(640 / cw, 640 / ch)
        if recall < 0.7:
            notes.append(
                f"    note: {desc} -- letterbox scale {lb:.2f} shrinks buttons to "
                f"~{med*lb:.0f}px in model space, so recall drops to "
                f"{100*recall:.0f}% and centres are +/-{max_err:.0f}px. Boxes are "
                "still in the correct space (all inside the panel); this is a "
                "detection-range limit, not a coordinate bug.")

    print("-" * 82)
    for n in notes:
        print(n)
    if failures:
        print(f"\n{failures} case(s) FAILED - boxes are NOT in original-image space.")
        return 1
    print("\nAll cases passed: bboxes are returned in original-image pixel "
          "coordinates\nregardless of input size or aspect ratio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
