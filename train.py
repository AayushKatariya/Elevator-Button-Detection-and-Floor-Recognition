#!/usr/bin/env python3
"""
Train YOLOv8n to detect elevator buttons.

Hyperparameters come from config.yaml so the README can't drift from what was
actually run. Writes reports/training_metadata.json with the hardware, wall
clock, and the epochs that really ran (early stopping often cuts it short).

Usage:
    python train.py
    python train.py --epochs 50 --device cpu
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def check_device(requested: str | None) -> str:
    """
    Pick the device, checking the GPU can actually run a kernel.

    is_available() is not enough. On a Blackwell card with a CUDA 12.4 torch it
    returns True and then every kernel launch fails, so Ultralytics picks the
    GPU and dies on the first forward pass with an error that points at CUDA
    rather than at the install.
    """
    import torch

    if requested == "cpu":
        print("  device: cpu (forced)")
        return "cpu"

    if not torch.cuda.is_available():
        print("  device: cpu (no CUDA device visible)")
        return "cpu"

    name = torch.cuda.get_device_name(0)
    try:
        x = torch.rand(512, 512, device="cuda")
        float((x @ x).sum())
    except RuntimeError as exc:
        print(f"\n  ERROR: CUDA reports available but cannot execute kernels on {name}.")
        print(f"         {str(exc).splitlines()[0]}")
        print(f"         torch {torch.__version__} supports {torch.cuda.get_arch_list()}")
        print("\n         This card needs a newer CUDA build. Fix with:")
        print("           pip install --force-reinstall torch torchvision \\")
        print("             --index-url https://download.pytorch.org/whl/cu128")
        print("\n         Or train on CPU (much slower):  python train.py --device cpu\n")
        sys.exit(2)

    print(f"  device: cuda:0 ({name}) - verified with a real matmul")
    return "0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, help="override config")
    ap.add_argument("--batch", type=int, help="override config")
    ap.add_argument("--imgsz", type=int, help="override config")
    ap.add_argument("--device", help="'cpu', '0', or omit to auto-detect")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg, paths = cfg["training"], cfg["paths"]
    for key in ("epochs", "batch", "imgsz"):
        if getattr(args, key) is not None:
            tcfg[key] = getattr(args, key)

    data_yaml = Path(paths["data_yaml"])
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run: python prepare_dataset.py", file=sys.stderr)
        return 1

    print("Environment")
    device = check_device(args.device)

    import torch
    from ultralytics import YOLO

    print(f"  torch  : {torch.__version__} (cuda build {torch.version.cuda})")
    print(f"  python : {platform.python_version()} on {platform.system()} {platform.release()}")

    dcfg = yaml.safe_load(data_yaml.read_text())
    n_train = len(list((Path(dcfg["path"]) / dcfg["train"]).glob("*")))
    n_val = len(list((Path(dcfg["path"]) / dcfg["val"]).glob("*")))
    print(f"\nDataset\n  {data_yaml}")
    print(f"  nc={dcfg['nc']} names={dcfg['names']}")
    print(f"  train={n_train}  val={n_val}")

    print(f"\nTraining  {tcfg['base_model']} -> {tcfg['epochs']} epochs "
          f"@ imgsz={tcfg['imgsz']} batch={tcfg['batch']} seed={tcfg['seed']}\n")

    model = YOLO(tcfg["base_model"])
    started = time.time()
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=tcfg["epochs"],
        imgsz=tcfg["imgsz"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        seed=tcfg["seed"],
        workers=tcfg["workers"],
        optimizer=tcfg["optimizer"],
        project=tcfg["project"],
        name=tcfg["name"],
        device=device,
        exist_ok=True,
        resume=args.resume,
        fliplr=tcfg["fliplr"], flipud=tcfg["flipud"],
        degrees=tcfg["degrees"], perspective=tcfg["perspective"],
        mosaic=tcfg["mosaic"],
        hsv_h=tcfg["hsv_h"], hsv_s=tcfg["hsv_s"], hsv_v=tcfg["hsv_v"],
    )
    elapsed = time.time() - started

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    dst = Path(paths["model_weights"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dst)
        print(f"\n  weights -> {dst}")
    else:
        print(f"\n  WARNING: {best} not found", file=sys.stderr)

    # What actually ran, not what was configured.
    epochs_run = tcfg["epochs"]
    csv = save_dir / "results.csv"
    if csv.exists():
        rows = [r for r in csv.read_text().splitlines() if r.strip()]
        epochs_run = max(0, len(rows) - 1)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
            "platform": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
        "dataset": {
            "data_yaml": str(data_yaml), "nc": dcfg["nc"], "names": dcfg["names"],
            "train_images": n_train, "val_images": n_val,
        },
        "config": tcfg,
        "epochs_configured": tcfg["epochs"],
        "epochs_run": epochs_run,
        "early_stopped": epochs_run < tcfg["epochs"],
        "wall_clock_seconds": round(elapsed, 1),
        "wall_clock_human": f"{int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s",
        "run_dir": str(save_dir),
        "weights": str(dst),
    }
    try:
        rd = results.results_dict
        meta["final_metrics"] = {k: (float(v) if v is not None else None) for k, v in rd.items()}
    except Exception:
        pass

    Path("reports").mkdir(exist_ok=True)
    Path("reports/training_metadata.json").write_text(json.dumps(meta, indent=1))

    print(f"""
  Done in {meta['wall_clock_human']}  ({epochs_run}/{tcfg['epochs']} epochs"""
          f"""{', early stopped' if meta['early_stopped'] else ''})

    weights   {dst}
    run dir   {save_dir}
    metadata  reports/training_metadata.json

  Next: python evaluate.py    (detection metrics on the test split)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
