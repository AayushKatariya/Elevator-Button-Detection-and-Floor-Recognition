#!/usr/bin/env python3
"""
Part 3 engine comparison.

Runs every available OCR engine over the SAME ground-truth button crops from the
test split, so the assignment's "Explain why you selected your approach" is
answered with measured numbers rather than an assertion.

Ground-truth boxes are used deliberately: this isolates recognition quality from
detection quality, and the resulting figure is the ceiling each engine imposes
on the full pipeline.

Usage:
    python ocr_compare.py
    python ocr_compare.py --limit 400 --engines easyocr,tesseract
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
from src.ocr import (EasyOcrEngine, PaddleOcrEngine,  # noqa: E402
                     TesseractEngine, snap_to_vocabulary)

def find_tesseract(cfg_path: str | None = None) -> str | None:
    """
    Locate the Tesseract binary: config value first, then PATH, then the usual
    install locations. Tesseract needs a SYSTEM binary, not just pytesseract,
    and hardcoding one machine's path would break a clean clone.
    """
    import shutil
    for cand in (cfg_path, shutil.which("tesseract")):
        if cand and Path(cand).exists():
            return cand
    for cand in ("C:/Program Files/Tesseract-OCR/tesseract.exe",
                 "/usr/bin/tesseract", "/usr/local/bin/tesseract",
                 "/opt/homebrew/bin/tesseract"):
        if Path(cand).exists():
            return cand
    return None


def size_bucket(px: float) -> str:
    if px < 24:  return "tiny(<24)"
    if px < 32:  return "small(24-32)"
    if px < 48:  return "med(32-48)"
    if px < 64:  return "large(48-64)"
    return "xl(>=64)"


def load_crops(cfg, split, limit):
    man = json.loads(Path(cfg["paths"]["ocr_ground_truth"]).read_text())
    root = Path(cfg["paths"]["dataset"])
    pad = cfg["ocr"]["crop_padding_ratio"]
    items = [(rel, b) for rel, e in man.items() if e["split"] == split
             for b in e["buttons"] if b["ocr_scorable"]]
    if limit:
        items = items[:limit]
    vocab = {b["text"] for e in man.values() for b in e["buttons"]
             if b["ocr_scorable"] and b["text"]}

    crops, cache = [], {}
    for rel, b in items:
        if rel not in cache:
            cache[rel] = cv2.imread(str(root / rel))
        img = cache[rel]
        if img is None:
            continue
        H, W = img.shape[:2]
        cx, cy, w, h = b["bbox_norm"]
        pw, ph = w * W * pad, h * H * pad
        x1 = int(max(0, (cx - w / 2) * W - pw)); y1 = int(max(0, (cy - h / 2) * H - ph))
        x2 = int(min(W, (cx + w / 2) * W + pw)); y2 = int(min(H, (cy + h / 2) * H + ph))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            continue
        side = float(np.sqrt(max(1.0, (w * W) * (h * H))))
        crops.append((img[y1:y2, x1:x2], b["text"], side))
    return crops, vocab


def evaluate(name, engine, crops, vocab, use_tta=True):
    n = correct = snapped_ok = no_read = 0
    by_size = defaultdict(lambda: [0, 0])
    by_len = defaultdict(lambda: [0, 0])
    conf = Counter()
    lat = []
    for crop, truth, side in crops:
        t0 = time.perf_counter()
        try:
            res = engine.read_crop(crop, use_tta=use_tta)
        except Exception:
            res = None
        lat.append((time.perf_counter() - t0) * 1000)
        got = res.text if res else None
        n += 1
        bs, bl = size_bucket(side), len(truth)
        by_size[bs][1] += 1
        by_len[bl][1] += 1
        if not got:
            no_read += 1
            conf[(truth, "<no read>")] += 1
            continue
        if got == truth:
            correct += 1; snapped_ok += 1
            by_size[bs][0] += 1; by_len[bl][0] += 1
        else:
            if snap_to_vocabulary(got, vocab, 1)[0] == truth:
                snapped_ok += 1
            conf[(truth, got)] += 1
    return {
        "engine": name, "evaluated": n, "correct": correct,
        "incorrect": n - correct - no_read, "no_read": no_read,
        "accuracy": round(correct / max(1, n), 4),
        "accuracy_snapped": round(snapped_ok / max(1, n), 4),
        "read_rate": round((n - no_read) / max(1, n), 4),
        "latency_ms": {"mean": round(float(np.mean(lat)), 1),
                       "median": round(float(np.median(lat)), 1)},
        "by_size": {k: {"correct": v[0], "total": v[1]} for k, v in by_size.items()},
        "by_label_len": {str(k): {"correct": v[0], "total": v[1]} for k, v in by_len.items()},
        "top_confusions": [{"truth": t, "got": g, "n": c} for (t, g), c in conf.most_common(20)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--engines", default="easyocr,tesseract,paddleocr")
    ap.add_argument("--out", default="reports/ocr_engine_comparison.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    o = cfg["ocr"]
    crops, vocab = load_crops(cfg, args.split, args.limit)
    print(f"OCR engine comparison -- ground-truth crops, split='{args.split}'")
    print(f"  crops={len(crops)}  vocab={len(vocab)}  padding={o['crop_padding_ratio']}\n")

    builders = {
        "easyocr": lambda: EasyOcrEngine(
            allowlist=o["allowlist"], min_confidence=o["min_confidence"],
            target_crop_height=o["target_crop_height"]),
        "tesseract": lambda: TesseractEngine(
            allowlist=o["allowlist"], min_confidence=o["min_confidence"],
            target_crop_height=o["target_crop_height"],
            binary_path=find_tesseract(o.get("tesseract_path"))),
        "paddleocr": lambda: PaddleOcrEngine(
            allowlist=o["allowlist"], min_confidence=o["min_confidence"],
            target_crop_height=o["target_crop_height"]),
    }

    results = []
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name not in builders:
            print(f"  skipping unknown engine '{name}'"); continue
        print(f"  running {name} ...", flush=True)
        try:
            eng = builders[name]()
            _ = getattr(eng, "reader", None) or getattr(eng, "ocr", None)
        except Exception as exc:
            print(f"    unavailable: {type(exc).__name__}: {str(exc)[:120]}")
            results.append({"engine": name, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        r = evaluate(name, eng, crops, vocab)
        results.append(r)
        print(f"    accuracy {100*r['accuracy']:.1f}%  read-rate {100*r['read_rate']:.1f}%  "
              f"{r['latency_ms']['median']} ms/crop")

    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 78)
    print(f"{'engine':<12}{'accuracy':>10}{'+vocab':>9}{'read rate':>11}{'ms/crop':>9}"
          f"{'1-char':>9}{'2-char':>9}")
    print("-" * 78)
    for r in sorted(ok, key=lambda x: -x["accuracy"]):
        c1 = r["by_label_len"].get("1", {"correct": 0, "total": 0})
        c2 = r["by_label_len"].get("2", {"correct": 0, "total": 0})
        p1 = 100 * c1["correct"] / max(1, c1["total"])
        p2 = 100 * c2["correct"] / max(1, c2["total"])
        print(f"{r['engine']:<12}{100*r['accuracy']:9.1f}%{100*r['accuracy_snapped']:8.1f}%"
              f"{100*r['read_rate']:10.1f}%{r['latency_ms']['median']:9.1f}"
              f"{p1:8.1f}%{p2:8.1f}%")
    print("=" * 78)

    if ok:
        print("\naccuracy by button size:")
        buckets = ["tiny(<24)", "small(24-32)", "med(32-48)", "large(48-64)", "xl(>=64)"]
        print(f"  {'engine':<12}" + "".join(f"{b:>14}" for b in buckets))
        for r in sorted(ok, key=lambda x: -x["accuracy"]):
            cells = ""
            for b in buckets:
                d = r["by_size"].get(b)
                cells += f"{(100*d['correct']/d['total']):13.1f}%" if d and d["total"] else f"{'-':>14}"
            print(f"  {r['engine']:<12}{cells}")

        best = max(ok, key=lambda x: x["accuracy"])
        print(f"\nbest general-purpose engine: {best['engine']} at {100*best['accuracy']:.1f}%")

    Path("reports").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"split": args.split, "crops": len(crops),
         "crop_padding_ratio": o["crop_padding_ratio"], "results": results}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
