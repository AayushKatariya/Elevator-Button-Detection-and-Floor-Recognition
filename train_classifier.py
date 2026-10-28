#!/usr/bin/env python3
"""
Part 3 option: "a lightweight vision/OCR model".

Trains YOLOv8n-cls -- a ~1.5M parameter classifier -- to read a button crop as
closed-set classification over the label strings present in training, instead of
decoding an arbitrary character sequence.

Why this should beat a general OCR engine here, in one line: a CRNN's recurrent
stage exists to exploit context between characters, and a 1-3 character button
label has none, while the engine has also never seen engraved metal or specular
glare in training. A classifier trained on 12,553 in-domain crops has neither
handicap.

This mirrors the OCR-RCNN approach that reports 79.6% character accuracy on this
same dataset (Zhu et al.), and it is only possible because the recovered class
dictionary labels every crop for free.

Honest limitation to report: closed-set means it CANNOT read a label absent from
training. On this test split that caps it at 99.2%.

Prerequisite:  python export_crops.py

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 80 --imgsz 96
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crops")
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "train").is_dir():
        print(f"ERROR: {data}/train not found. Run: python export_crops.py")
        return 1

    import torch
    from ultralytics import YOLO

    device = args.device
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            try:
                x = torch.rand(256, 256, device="cuda"); float((x @ x).sum())
                device = "0"
            except RuntimeError:
                device = "cpu"
    print(f"device: {device}")

    n_classes = len([p for p in (data / "train").iterdir() if p.is_dir()])
    n_train = sum(1 for _ in (data / "train").rglob("*.png"))
    print(f"classes: {n_classes}   train crops: {n_train}\n")

    model = YOLO(args.model)
    t0 = time.time()
    results = model.train(
        data=str(data.resolve()),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, device=device,
        project="runs_cls", name="button_cls", exist_ok=True,
        # Buttons are never upside down and mirroring changes digit identity,
        # so BOTH flips stay off here -- unlike the detector, where the
        # single-class collapse made horizontal flip safe.
        fliplr=0.0, flipud=0.0,
        degrees=8.0, translate=0.08, scale=0.25,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        erasing=0.0,
    )
    elapsed = time.time() - t0

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    dst = Path("models/button_cls.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dst)

    top1 = top5 = None
    try:
        top1 = float(results.top1); top5 = float(results.top5)
    except Exception:
        pass

    Path("reports").mkdir(exist_ok=True)
    Path("reports/classifier_training.json").write_text(json.dumps({
        "model": args.model, "classes": n_classes, "train_crops": n_train,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "seed": args.seed, "device": device,
        "val_top1": top1, "val_top5": top5,
        "wall_clock_seconds": round(elapsed, 1),
        "weights": str(dst), "run_dir": str(save_dir),
    }, indent=1))

    print(f"""
  trained in {int(elapsed//60)}m {int(elapsed%60)}s
    val top-1 : {top1}
    val top-5 : {top5}
    weights   : {dst}

  Next: python eval_classifier.py    (test-split accuracy, comparable to the OCR engines)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
