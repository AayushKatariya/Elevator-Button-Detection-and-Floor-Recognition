#!/usr/bin/env python3
"""
Phase 2.5 -- OCR ceiling gate.

Runs OCR on GROUND-TRUTH button crops from the test split. No detector is
involved, so every box is perfect. Whatever accuracy comes out is the upper
bound the full pipeline can ever reach: detection errors, crop jitter and
threshold tuning can only take it down from here.

This is worth knowing before the OCR engine choice is locked, because 31% of
buttons in this dataset are under 32x32 px and no engine recovers detail that
was never captured. If the ceiling is low, grid inference in the floor selector
stops being a nice-to-have and becomes load-bearing.

Usage:
    python ocr_gate.py
    python ocr_gate.py --limit 300 --no-tta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.ocr import EasyOcrEngine, crop_button, snap_to_vocabulary  # noqa: E402


def bucket(px: float) -> str:
    if px < 24:   return "tiny   (<24px)"
    if px < 32:   return "small  (24-32px)"
    if px < 48:   return "medium (32-48px)"
    if px < 64:   return "large  (48-64px)"
    return "xlarge (>=64px)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, help="cap number of buttons (for a quick pass)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="reports/ocr_ceiling.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gt_path = Path(cfg["paths"]["ocr_ground_truth"])
    root = Path(cfg["paths"]["dataset"])
    manifest = json.loads(gt_path.read_text())

    # Collect every OCR-scorable button in the chosen split.
    items = []
    for rel, entry in manifest.items():
        if entry["split"] != args.split:
            continue
        for b in entry["buttons"]:
            if b["ocr_scorable"]:
                items.append((rel, b))
    if args.limit:
        items = items[: args.limit]

    vocab = {b["text"] for e in manifest.values() for b in e["buttons"]
             if b["ocr_scorable"] and b["text"]}

    print(f"OCR ceiling gate -- ground-truth crops, split='{args.split}'")
    print(f"  buttons to evaluate : {len(items)}")
    print(f"  known label vocab   : {len(vocab)} strings")
    print(f"  TTA                 : {not args.no_tta}")

    ocfg = cfg["ocr"]
    engine = EasyOcrEngine(
        allowlist=ocfg["allowlist"],
        min_confidence=ocfg["min_confidence"],
        target_crop_height=ocfg["target_crop_height"],
        gpu=not args.cpu,
    )
    print("  loading EasyOCR (first run downloads ~100MB) ...")
    _ = engine.reader
    print("  ready\n")

    by_image = defaultdict(list)
    for rel, b in items:
        by_image[rel].append(b)

    n = correct = correct_snapped = no_read = 0
    by_size = defaultdict(lambda: [0, 0])
    confusion = Counter()
    latencies = []
    started = time.time()

    for i, (rel, buttons) in enumerate(sorted(by_image.items()), 1):
        img = cv2.imread(str(root / rel))
        if img is None:
            continue
        H, W = img.shape[:2]
        for b in buttons:
            cx, cy, w, h = b["bbox_norm"]
            xyxy = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            crop = crop_button(img, xyxy, ocfg["crop_padding_ratio"])

            t0 = time.perf_counter()
            res = engine.read_crop(crop, use_tta=not args.no_tta)
            latencies.append((time.perf_counter() - t0) * 1000)

            truth = b["text"]
            got = res.text
            side = float(np.sqrt(max(1.0, (w * W) * (h * H))))
            bk = bucket(side)

            n += 1
            by_size[bk][1] += 1
            if got is None or got == "":
                no_read += 1
                confusion[(truth, "<no read>")] += 1
                continue
            if got == truth:
                correct += 1
                correct_snapped += 1
                by_size[bk][0] += 1
            else:
                snapped, _ = snap_to_vocabulary(got, vocab, 1)
                if snapped == truth:
                    correct_snapped += 1
                confusion[(truth, got)] += 1

        if i % 25 == 0 or i == len(by_image):
            pct = 100 * correct / max(1, n)
            print(f"  {i}/{len(by_image)} images | {n} buttons | exact {pct:.1f}%")

    elapsed = time.time() - started
    wrong = n - correct - no_read

    print(f"""
{'='*62}
OCR CEILING  (ground-truth boxes -- the pipeline can never beat this)
{'='*62}
  Number of buttons evaluated : {n}
  Correctly recognized        : {correct}
  Incorrectly recognized      : {wrong}
  No read returned            : {no_read}

  Exact-match accuracy        : {100*correct/max(1,n):.1f}%
  With vocabulary snapping    : {100*correct_snapped/max(1,n):.1f}%
  Read-attempt rate           : {100*(n-no_read)/max(1,n):.1f}%

  Latency per crop            : mean {np.mean(latencies):.1f} ms | median {np.median(latencies):.1f} ms
  Total wall clock            : {elapsed:.1f}s
""")

    print("  Accuracy by button size (this is the story):")
    for k in ["tiny   (<24px)", "small  (24-32px)", "medium (32-48px)",
              "large  (48-64px)", "xlarge (>=64px)"]:
        ok, tot = by_size[k]
        if tot:
            print(f"    {k:20s} {ok:5d}/{tot:<5d}  {100*ok/tot:5.1f}%")

    print("\n  Most common misreads (truth -> got):")
    for (t, g), c in confusion.most_common(15):
        print(f"    {t:>6s} -> {g:<12s} {c}")

    Path("reports").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "split": args.split, "tta": not args.no_tta,
        "evaluated": n, "correct": correct, "incorrect": wrong, "no_read": no_read,
        "exact_accuracy": round(correct / max(1, n), 4),
        "snapped_accuracy": round(correct_snapped / max(1, n), 4),
        "latency_ms": {"mean": round(float(np.mean(latencies)), 2),
                       "median": round(float(np.median(latencies)), 2),
                       "p95": round(float(np.percentile(latencies, 95)), 2)},
        "by_size": {k: {"correct": v[0], "total": v[1]} for k, v in by_size.items()},
        "confusion": [{"truth": t, "got": g, "count": c}
                      for (t, g), c in confusion.most_common(60)],
    }, indent=1))
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
