#!/usr/bin/env python3
"""
Detect and read every button on a panel.

    python inference.py --image panel.jpg
    python inference.py --image panel.jpg --visualize
    python inference.py --dir examples/input --visualize --json-out examples/output

Use detect_floor.py when you only want one specific floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect and read all elevator buttons.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--dir", help="a directory of panel images")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--out-dir", default="examples/output")
    ap.add_argument("--json-out", help="directory to write one JSON per image")
    ap.add_argument("--device")
    ap.add_argument("--quiet", action="store_true", help="JSON only, no table")
    args = ap.parse_args()

    from src.pipeline import ElevatorPipeline, load_config, write_json

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(p for p in Path(args.dir).iterdir()
                       if p.suffix.lower() in IMAGE_EXT)
    if not paths:
        print("error: no images found", file=sys.stderr)
        return 1

    pipe = ElevatorPipeline(load_config(args.config), device=args.device)
    results = []

    for path in paths:
        if not path.exists():
            print(f"skipping missing {path}", file=sys.stderr)
            continue

        image = pipe.imread(path)
        dets = pipe.run(image)
        result = pipe.result_json(dets, image_path=str(path))
        results.append(result)

        if not args.quiet:
            t = pipe.timings
            read = sum(1 for d in dets if d.text)
            print(f"\n{path.name}   {image.shape[1]}x{image.shape[0]}   "
                  f"{len(dets)} buttons, {read} read   "
                  f"[detect {t.get('detect_ms', 0):.0f}ms, ocr {t.get('ocr_ms', 0):.0f}ms]")
            print(f"  {'#':>3}  {'text':<6} {'bbox':<26} {'center':<14} {'det':>5} {'ocr':>5}")
            for i, d in enumerate(dets, 1):
                bbox = ",".join(str(v) for v in d.bbox)
                ctr = ",".join(str(v) for v in d.center)
                mark = "*" if d.inferred else " "
                print(f"  {i:>3}  {(d.text or '?'):<5}{mark} [{bbox:<24}] "
                      f"({ctr:<12}) {d.confidence:5.2f} {d.ocr_confidence:5.2f}")

        if args.visualize:
            import cv2
            from src.viz import annotate, legend
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{path.stem}_annotated.jpg"
            cv2.imwrite(str(dst), legend(annotate(image, dets)))
            print(f"  annotated -> {dst}", file=sys.stderr)

        if args.json_out:
            write_json(result, Path(args.json_out) / f"{path.stem}.json")

    if args.quiet:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
