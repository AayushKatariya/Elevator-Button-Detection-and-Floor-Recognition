#!/usr/bin/env python3
"""
Given a panel image and a target floor, print the button to press.

    python detect_floor.py --image panel.jpg --floor 27

    {
      "target_floor": 27,
      "button": {
        "text": "27",
        "bbox": [412, 328, 468, 382],
        "center": [440, 355],
        "confidence": 0.94
      }
    }

Only that JSON goes to stdout; logs and warnings go to stderr, so this pipes
cleanly into something else. bbox is [x1, y1, x2, y2] in pixels of the original
image, whatever its size.

Exit codes: 0 found, 3 no matching button, 1 bad input.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

# Set before the heavy imports read them.
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("KMP_WARNINGS", "0")

sys.path.insert(0, str(Path(__file__).parent))


@contextlib.contextmanager
def quiet_stdout():
    """
    Point stdout at stderr for the duration, at the file-descriptor level.

    redirect_stdout alone isn't enough: PaddleOCR logs from C++ straight to
    fd 1, which lands in the middle of our JSON and makes it unparseable.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main() -> int:
    ap = argparse.ArgumentParser(description="Find the button for a target floor.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--floor", required=True, help="e.g. 27, B1, G, LG2")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--visualize", metavar="PATH", nargs="?", const="auto",
                    help="write an annotated image (default: examples/output/)")
    ap.add_argument("--all", action="store_true",
                    help="also list every detected button")
    ap.add_argument("--device", help="'cpu' or a CUDA index; default auto")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"error: no such image: {img_path}", file=sys.stderr)
        return 1

    with quiet_stdout():
        import warnings
        warnings.filterwarnings("ignore")
        from src.pipeline import ElevatorPipeline, load_config

        cfg = load_config(args.config)
        pipe = ElevatorPipeline(cfg, device=args.device)
        image = pipe.imread(img_path)
        selection, dets = pipe.find_floor(image, args.floor)

    result = selection.to_dict(cfg["floor_selection"]["confidence_mode"])
    if args.all:
        result["all_buttons"] = [d.to_dict() for d in dets]

    if args.visualize:
        import cv2
        from src.viz import annotate, legend
        out_path = (Path("examples/output") / f"{img_path.stem}_floor{args.floor}.jpg"
                    if args.visualize == "auto" else Path(args.visualize))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), legend(annotate(image, dets, selection.button)))
        print(f"annotated -> {out_path}", file=sys.stderr)

    print(json.dumps(result, indent=2))

    if selection.button is None:
        print(f"no button matched floor {args.floor!r} ({selection.reason}); "
              f"{len(dets)} button(s) detected", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
