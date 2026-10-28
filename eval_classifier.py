#!/usr/bin/env python3
"""
Score the button classifier on the test split, comparably to the OCR engines.

Ultralytics' built-in classification val() is NOT usable here: it derives the
class list from the directory listing but loads samples with ImageFolder, which
skips empty directories. Because rare labels do not appear in every split
(228 train / 126 val / 106 test classes), its label->index mapping differs per
split and the reported accuracy is meaningless -- it printed 11% top-1 for a
model that is actually far better.

This script sidesteps that entirely: it predicts each test crop and maps the
predicted index through the MODEL'S OWN names dict back to a label string, then
compares strings. Same crops, same metric, same denominator as ocr_compare.py.

Usage:
    python eval_classifier.py
    python eval_classifier.py --weights models/button_cls.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def unsafe_name(dirname: str) -> str:
    """Invert export_crops.safe_name(): 'cls_neg1' -> '-1', 'cls_B1' -> 'B1'."""
    s = re.sub(r"^cls_", "", dirname)
    return s.replace("neg", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/button_cls.pt")
    ap.add_argument("--crops", default="crops/test")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="reports/classifier_eval.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = {int(k): unsafe_name(v) for k, v in model.names.items()}
    print(f"classifier: {args.weights}   classes: {len(names)}")

    files, truths = [], []
    for d in sorted(Path(args.crops).iterdir()):
        if not d.is_dir():
            continue
        label = unsafe_name(d.name)
        for f in sorted(d.glob("*.png")):
            files.append(str(f)); truths.append(label)
    print(f"test crops: {len(files)}\n")
    if not files:
        print("ERROR: no test crops. Run: python export_crops.py")
        return 1

    known = set(names.values())
    unseen = sum(1 for t in truths if t not in known)

    preds, confs = [], []
    t0 = time.perf_counter()
    for i in range(0, len(files), args.batch):
        for r in model.predict(files[i:i + args.batch], verbose=False):
            top = int(r.probs.top1)
            preds.append(names.get(top, "?"))
            confs.append(float(r.probs.top1conf))
    ms = (time.perf_counter() - t0) * 1000 / len(files)

    n = len(truths)
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    by_len = defaultdict(lambda: [0, 0])
    conf_tab = Counter()
    for p, t in zip(preds, truths):
        by_len[len(t)][1] += 1
        if p == t:
            by_len[len(t)][0] += 1
        else:
            conf_tab[(t, p)] += 1

    # accuracy at a confidence gate, mirroring the OCR engines' min_confidence
    gated = [(p, t, c) for p, t, c in zip(preds, truths, confs) if c >= 0.30]
    gated_ok = sum(1 for p, t, _ in gated if p == t)

    print(f"{'='*58}")
    print("BUTTON CLASSIFIER  (YOLOv8n-cls, ground-truth crops, test split)")
    print(f"{'='*58}")
    print(f"  Number of buttons evaluated : {n}")
    print(f"  Correctly recognized        : {correct}")
    print(f"  Incorrectly recognized      : {n - correct}")
    print(f"  Accuracy                    : {100*correct/n:.1f}%")
    print(f"  Accuracy @conf>=0.30        : {100*gated_ok/max(1,len(gated)):.1f}%"
          f"  (on {len(gated)} of {n} = {100*len(gated)/n:.1f}% answered)")
    print(f"  Unreadable by construction  : {unseen} ({100*unseen/n:.1f}%) label never seen in training")
    print(f"  Latency                     : {ms:.2f} ms/crop\n")

    print("  accuracy by label length:")
    for k, v in sorted(by_len.items()):
        if v[1]:
            print(f"    {k}-char: {v[0]:5d}/{v[1]:<5d} {100*v[0]/v[1]:5.1f}%")

    print("\n  most common confusions (truth -> got):")
    for (t, p), c in conf_tab.most_common(12):
        print(f"    {t:>6s} -> {p:<8s} {c}")

    Path("reports").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "weights": args.weights, "classes": len(names), "evaluated": n,
        "correct": correct, "accuracy": round(correct / n, 4),
        "accuracy_gated_0.30": round(gated_ok / max(1, len(gated)), 4),
        "answered_at_gate": len(gated),
        "unseen_label_count": unseen,
        "ms_per_crop": round(ms, 3),
        "by_label_len": {str(k): {"correct": v[0], "total": v[1]}
                         for k, v in sorted(by_len.items())},
        "top_confusions": [{"truth": t, "got": p, "n": c}
                           for (t, p), c in conf_tab.most_common(25)],
    }, indent=1))
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
