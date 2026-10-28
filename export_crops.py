#!/usr/bin/env python3
"""
Export per-button crops as a classification dataset.

Both remaining Part 3 options -- "OpenCV-based methods" and "a lightweight
vision/OCR model" -- treat button reading as CLOSED-SET CLASSIFICATION rather
than open-vocabulary OCR: pick one of the label strings seen in training,
instead of decoding an arbitrary character sequence.

That reframing is what the OCR-RCNN line of work does for this exact dataset
(Zhu et al. report 79.6% character accuracy on it), and it is only possible
because the recovered class dictionary gives us a correct label for all 20,513
boxes for free.

The trade-off is real and must be reported: a classifier CANNOT read a label it
never saw in training, whereas an OCR engine can. This script measures exactly
how many test buttons fall in that gap.

Crop geometry matches the tuned OCR settings (pad -0.22, INTER_LINEAR) so the
comparison against the OCR engines is like-for-like.

Usage:
    python export_crops.py
    python export_crops.py --size 64 --out crops
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}


def safe_name(label: str) -> str:
    """Filesystem-safe class directory name ('-1' -> 'neg1', '/' etc. stripped)."""
    s = label.replace("-", "neg")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return f"cls_{s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--size", type=int, default=64, help="output crop size (square)")
    ap.add_argument("--padding", type=float, default=None, help="default: config value")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pad = args.padding if args.padding is not None else cfg["ocr"]["crop_padding_ratio"]
    man = json.loads(Path(cfg["paths"]["ocr_ground_truth"]).read_text())
    root = Path(cfg["paths"]["dataset"])
    out = Path(args.out)

    if out.exists():
        shutil.rmtree(out)

    counts = {s: Counter() for s in SPLIT_DIR}
    arrays = {s: {"X": [], "y": []} for s in SPLIT_DIR}
    written = 0

    print(f"Exporting crops  size={args.size}px  padding={pad}")
    for i, (rel, entry) in enumerate(sorted(man.items()), 1):
        split = entry["split"]
        img = cv2.imread(str(root / rel))
        if img is None:
            continue
        H, W = img.shape[:2]
        for j, b in enumerate(entry["buttons"]):
            if not b["ocr_scorable"]:
                continue
            label = b["text"]
            cx, cy, w, h = b["bbox_norm"]
            pw, ph = w * W * pad, h * H * pad
            x1 = int(max(0, (cx - w / 2) * W - pw)); y1 = int(max(0, (cy - h / 2) * H - ph))
            x2 = int(min(W, (cx + w / 2) * W + pw)); y2 = int(min(H, (cy + h / 2) * H + ph))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue
            crop = cv2.resize(img[y1:y2, x1:x2], (args.size, args.size),
                              interpolation=cv2.INTER_LINEAR)

            d = out / SPLIT_DIR[split] / safe_name(label)
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{Path(rel).stem}_{j}.png"), crop)
            counts[split][label] += 1
            arrays[split]["X"].append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            arrays[split]["y"].append(label)
            written += 1
        if i % 400 == 0:
            print(f"  {i}/{len(man)} images  {written} crops")

    # NOTE on Ultralytics' classification loader: it derives the class list from
    # the directory listing but builds samples with torchvision ImageFolder,
    # which SKIPS EMPTY DIRECTORIES. Rare labels do not occur in every split, so
    # train/val/test legitimately expose different class sets (228/126/106) and
    # Ultralytics' own val number is not trustworthy here -- it reported 11%
    # top-1 on a model that actually works. Padding the splits with empty
    # directories does NOT fix it (they are skipped).
    #
    # So: leave the directories as they are, train on the training vocabulary,
    # and score the test split in eval_classifier.py using the model's own
    # names mapping. That is the number comparable to the OCR engines.
    for s in SPLIT_DIR:                       # drop any empty dirs left behind
        for d in (out / SPLIT_DIR[s]).iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # Save arrays for the classical OpenCV baselines (no disk re-read needed).
    np.savez_compressed(
        out / "arrays.npz",
        **{f"{s}_X": np.array(arrays[s]["X"], np.uint8) for s in SPLIT_DIR},
        **{f"{s}_y": np.array(arrays[s]["y"], object) for s in SPLIT_DIR},
    )

    train_labels = set(counts["train"])
    test_total = sum(counts["test"].values())
    unseen = {k: v for k, v in counts["test"].items() if k not in train_labels}

    print(f"""
  crops written : {written}
  train / val / test : {sum(counts['train'].values())} / {sum(counts['valid'].values())} / {test_total}
  distinct labels    : train {len(counts['train'])}  test {len(counts['test'])}

  CLOSED-SET CEILING
    test buttons whose label never appears in train : {sum(unseen.values())} / {test_total}"""
          f""" ({100*sum(unseen.values())/max(1,test_total):.1f}%)
    -> a classifier can score at most {100*(1-sum(unseen.values())/max(1,test_total)):.1f}% on this test set
    unseen labels: {sorted(unseen)[:18]}
""")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/crop_dataset.json").write_text(json.dumps({
        "size": args.size, "padding": pad, "total": written,
        "per_split": {s: sum(counts[s].values()) for s in SPLIT_DIR},
        "distinct_labels": {s: len(counts[s]) for s in SPLIT_DIR},
        "unseen_in_train": {"count": sum(unseen.values()),
                            "fraction": round(sum(unseen.values()) / max(1, test_total), 4),
                            "labels": sorted(unseen)},
        "train_label_counts": dict(counts["train"].most_common()),
    }, indent=1))
    print(f"  -> {out}/  and reports/crop_dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
