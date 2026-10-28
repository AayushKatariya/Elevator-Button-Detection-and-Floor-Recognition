#!/usr/bin/env python3
"""
Build the merged elevator-button detection dataset.

Dataset/ and Dataset 2/ are two exports of the same 2019 photographs, and
neither is usable alone. Dataset/ has 640x640 images with ~1.5x the real detail
but a corrupt data.yaml (the class names are lines of README prose) and seg
polygons instead of boxes. Dataset 2/ is only 416x416 but has clean bbox labels
and a working 368-class dictionary where the class name IS the button text.

So: Dataset/'s pixels, Dataset 2/'s labels. Out comes a single-class detection
dataset plus an OCR ground-truth manifest keeping the original text per box.

Images pair by filename stem (Roboflow appends a fresh .rf.<hash> per export, so
full names never match), and every pair is checked by pixel similarity -- a bad
pairing should fail loudly, not quietly put boxes on the wrong photo.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --out "Final dataset" --seed 42 --ratios 0.7 0.2 0.1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# --- class policy ---------------------------------------------------------
# Every call below came from looking at crops of that class, not from guessing
# at the name. The README has the reasoning and the counts.

# Panel elements that are NOT pressable buttons. Dropped entirely: they are not
# detection targets, and several are actively harmful if kept (a seven-segment
# floor display or a printed "OPEN DOOR" placard would be OCR'd as a button
# label and could send the robot to a floor that has no button).
NON_BUTTON = {
    "led",        # seven-segment floor indicator displays
    "indicator",  # illuminated status lamps
    "light",      # lamps
    "text",       # printed placards beside buttons
    "speaker",    # speaker grilles
    "hat",        # emblems / logos
    "keyhole",    # key cylinders - not pressable
    "bt_keyhole",
    "key",
    "switch",     # toggle / rotary switches - not pressable buttons
    "bt_switch",
}
# All "text_*" classes (printed labels such as text_DOOR OPEN) are dropped too;
# handled by prefix in classify() below.

# Real buttons whose face carries no readable alphanumeric text. They stay in
# the detector's training data -- a button detector must find them -- but they
# are excluded from the OCR accuracy denominator, because there is no text to
# read and scoring them as OCR failures would understate the engine.
NO_TEXT_BUTTON = {
    "empty",    # pressable but unmarked / label not visible
    "blur",     # genuine button, out of focus
    "unknown",  # genuine button, label unrecognisable
    "alarm", "call", "open", "close", "up", "down", "updown",
    "stop", "fan", "fire",
    # Pictogram buttons whose CLASS NAME is a mnemonic, not text printed on the
    # button. Verified by inspecting crops. An OCR engine can never return "CH"
    # for a wheelchair icon, so scoring these would silently deflate accuracy.
    "CH",   # wheelchair / accessibility symbol
    "s",    # bare star glyph (as a prefix, 's' still means star + a letter)
    "-",    # dash / bar glyph
}

DROP, TEXT, NOTEXT = "drop", "text_button", "symbol_button"


def classify(name: str) -> str:
    """Return DROP / TEXT / NOTEXT for a Dataset-2 class name."""
    if name.startswith("text_") or name in NON_BUTTON:
        return DROP
    if name in NO_TEXT_BUTTON:
        return NOTEXT
    return TEXT


def ocr_text(name: str) -> str:
    """
    Ground-truth string an OCR engine should return for this button.

    A leading lowercase 's' marks a star glyph printed beside the label
    (e.g. 'sG' renders as a star followed by G). OCR reads the character, not
    the star, so the star prefix is stripped.
    """
    if len(name) > 1 and name[0] == "s" and not name[1].islower():
        return name[1:]
    return name


# --- helpers --------------------------------------------------------------

STEM_RE = re.compile(r"\.rf\.[0-9a-f]+\.\w+$")


def stem_of(path: Path) -> str:
    """Strip Roboflow's per-export '.rf.<hash>.<ext>' suffix."""
    return STEM_RE.sub("", path.name)


def signature(path: Path, size: int) -> np.ndarray:
    """Mean-centred, L2-normalised grayscale thumbnail. Cosine-comparable."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), np.float32)
    a = a.ravel() - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def index_images(root: Path) -> dict[str, Path]:
    out = {}
    for split in ("train", "valid", "test"):
        for p in sorted((root / split / "images").glob("*")):
            out[stem_of(p)] = p
    return out


def index_labels(root: Path) -> dict[str, Path]:
    out = {}
    for split in ("train", "valid", "test"):
        for p in sorted((root / split / "labels").glob("*.txt")):
            out[stem_of(p)] = p
    return out


def connected_components(adj: dict[int, set[int]], n: int) -> list[list[int]]:
    """Union the adjacency map into groups; every node appears exactly once."""
    seen, groups = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            seen.add(u)
            stack.extend(adj.get(u, set()) - comp)
        groups.append(sorted(comp))
    return groups


# --- main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-from", default="Dataset",
                    help="Dataset 1 root - supplies the 640px image files")
    ap.add_argument("--labels-from", default="Dataset 2",
                    help="Dataset 2 root - supplies bbox labels and the class dictionary")
    ap.add_argument("--out", default="Final dataset", help="output dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.20, 0.10],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--pair-threshold", type=float, default=0.99,
                    help="min cosine similarity between paired D1/D2 images")
    ap.add_argument("--dup-threshold", type=float, default=0.95,
                    help="cosine similarity above which two images are near-duplicates")
    ap.add_argument("--keep-empty-images", action="store_true",
                    help="keep images left with zero boxes after filtering")
    args = ap.parse_args()

    d1_root, d2_root, out_root = Path(args.images_from), Path(args.labels_from), Path(args.out)
    for p in (d1_root, d2_root):
        if not p.is_dir():
            print(f"ERROR: not a directory: {p}", file=sys.stderr)
            return 1

    # ---------------------------------------------------------------- 1. index
    print("[1/7] Indexing both exports ...")
    d1_images = index_images(d1_root)
    d2_images = index_images(d2_root)
    d2_labels = index_labels(d2_root)

    names = yaml.safe_load((d2_root / "data.yaml").read_text())["names"]
    stems = sorted(set(d1_images) & set(d2_labels) & set(d2_images))

    print(f"      Dataset 1 images : {len(d1_images)}")
    print(f"      Dataset 2 labels : {len(d2_labels)}")
    print(f"      paired stems     : {len(stems)}")
    only1, only2 = set(d1_images) - set(d2_labels), set(d2_labels) - set(d1_images)
    if only1 or only2:
        print(f"      WARNING unpaired -> only in D1: {len(only1)}, only in D2: {len(only2)}")
    if not stems:
        print("ERROR: no paired images found.", file=sys.stderr)
        return 1

    # ------------------------------------------------- 2. assert the pairing
    # Filename conventions break quietly. Prove per image that the D1 file and
    # the D2 file whose labels we are about to attach are the same photograph.
    print(f"[2/7] Verifying pairing by pixel similarity (threshold {args.pair_threshold}) ...")
    sigs64: dict[str, np.ndarray] = {}
    failures = []
    for i, st in enumerate(stems, 1):
        a = signature(d1_images[st], 64)
        b = signature(d2_images[st], 64)
        sim = float(a @ b)
        sigs64[st] = a
        if sim < args.pair_threshold:
            failures.append((st, sim))
        if i % 500 == 0 or i == len(stems):
            print(f"      {i}/{len(stems)}")
    if failures:
        print(f"\nERROR: {len(failures)} pair(s) failed the similarity assertion:", file=sys.stderr)
        for st, sim in failures[:20]:
            print(f"       {st}: similarity {sim:.4f}", file=sys.stderr)
        print("Refusing to emit a dataset with boxes on the wrong photos.", file=sys.stderr)
        return 2
    print(f"      all {len(stems)} pairs verified as the same photograph")

    # ------------------------------------------- 3. read + filter the labels
    print("[3/7] Reading labels, dropping non-button classes ...")
    boxes: dict[str, list[tuple[float, float, float, float, str]]] = {}
    kept_by_name, dropped_by_name = Counter(), Counter()
    total_in = 0

    for st in stems:
        rows = []
        for line in d2_labels[st].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"ERROR: {d2_labels[st]} is not YOLOv8 bbox format "
                      f"({len(parts)} fields on a line).", file=sys.stderr)
                return 3
            total_in += 1
            name = names[int(parts[0])]
            cx, cy, w, h = (float(v) for v in parts[1:])
            kind = classify(name)
            if kind is DROP:
                dropped_by_name[name] += 1
                continue
            # clamp to the image and discard anything degenerate
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
            if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
                dropped_by_name[f"<degenerate:{name}>"] += 1
                continue
            rows.append((((x1 + x2) / 2), ((y1 + y2) / 2), x2 - x1, y2 - y1, name))
            kept_by_name[name] += 1
        boxes[st] = rows

    kept = sum(len(v) for v in boxes.values())
    print(f"      instances in  : {total_in}")
    print(f"      kept          : {kept}")
    print(f"      dropped       : {total_in - kept}")

    empty_stems = [s for s in stems if not boxes[s]]
    if empty_stems and not args.keep_empty_images:
        print(f"      removing {len(empty_stems)} image(s) left with zero buttons")
        stems = [s for s in stems if boxes[s]]

    # ------------------------------------------------ 4. near-duplicate groups
    # Two exports of this data ship 4 duplicate pairs that straddle the original
    # train/test boundary. Group duplicates so a split can never separate them.
    print(f"[4/7] Clustering near-duplicates (threshold {args.dup_threshold}) ...")
    order = list(stems)
    sig_matrix = np.stack([signature(d1_images[s], 32) for s in order])
    sim = sig_matrix @ sig_matrix.T
    np.fill_diagonal(sim, -1.0)

    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in np.argwhere(sim > args.dup_threshold):
        if i < j:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
    groups = connected_components(adj, len(order))
    dup_groups = [g for g in groups if len(g) > 1]
    print(f"      {len(groups)} groups covering {len(order)} images "
          f"({len(dup_groups)} contain near-duplicates)")

    # ---------------------------------------------------------- 5. split
    # Split at GROUP level, so near-duplicates always land in the same split.
    print(f"[5/7] Splitting {args.ratios} with seed {args.seed} ...")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(groups))
    tr, va, _ = args.ratios
    n_train, n_valid = int(round(len(order) * tr)), int(round(len(order) * va))

    assign: dict[str, str] = {}
    n = 0
    for gi in perm:
        split = "train" if n < n_train else "valid" if n < n_train + n_valid else "test"
        for idx in groups[gi]:
            assign[order[idx]] = split
        n += len(groups[gi])

    counts = Counter(assign.values())
    print("      " + "  ".join(f"{k}={counts[k]}" for k in ("train", "valid", "test")))

    # ---------------------------------------------------------- 6. write out
    print(f"[6/7] Writing '{out_root}' ...")
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "valid", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for st in stems:
        split = assign[st]
        src = d1_images[st]
        dst_img = out_root / split / "images" / f"{st}{src.suffix}"
        shutil.copy2(src, dst_img)

        # Single-class collapse: every real button becomes class 0.
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h, _ in boxes[st]]
        (out_root / split / "labels" / f"{st}.txt").write_text("\n".join(lines) + "\n")

        # OCR ground truth keeps the ORIGINAL button text for every box, which
        # is what makes the OCR and end-to-end reports auto-gradeable.
        manifest[f"{split}/images/{dst_img.name}"] = {
            "split": split,
            "source_image": str(src).replace("\\", "/"),
            "source_label": str(d2_labels[st]).replace("\\", "/"),
            "buttons": [
                {
                    "bbox_norm": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
                    "class_name": name,
                    "text": ocr_text(name) if classify(name) is TEXT else None,
                    "ocr_scorable": classify(name) is TEXT,
                }
                for cx, cy, w, h, name in boxes[st]
            ],
        }

    (out_root / "data.yaml").write_text(
        "# Merged elevator-button dataset\n"
        "#   images : Dataset 1 (640x640, sharper)\n"
        "#   labels : Dataset 2 (clean bbox + authoritative class dictionary)\n"
        "# Detection is single-class; per-button text lives in ocr_ground_truth.json\n"
        f"path: {out_root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['button']\n"
    )
    (out_root / "ocr_ground_truth.json").write_text(json.dumps(manifest, indent=1))

    # ---------------------------------------------------------- 7. report
    print("[7/7] Writing report ...")
    scorable = sum(1 for v in manifest.values() for b in v["buttons"] if b["ocr_scorable"])
    per_split_inst = Counter()
    for v in manifest.values():
        per_split_inst[v["split"]] += len(v["buttons"])

    report = {
        "sources": {
            "images_from": str(d1_root), "labels_from": str(d2_root),
            "paired_images": len(stems),
        },
        "pairing_verification": {
            "method": "64x64 mean-centred grayscale cosine similarity, per image",
            "threshold": args.pair_threshold, "failures": 0,
        },
        "instances": {
            "input": total_in, "kept": kept, "dropped": total_in - kept,
            "ocr_scorable": scorable,
            "symbol_or_unreadable": kept - scorable,
        },
        "split": {
            "seed": args.seed, "ratios": args.ratios,
            "images": dict(counts), "instances": dict(per_split_inst),
            "duplicate_groups_kept_together": len(dup_groups),
            "duplicate_threshold": args.dup_threshold,
        },
        "dropped_classes": dict(dropped_by_name.most_common()),
        "kept_classes": dict(kept_by_name.most_common()),
        "duplicate_groups": [[order[i] for i in g] for g in dup_groups],
    }
    (out_root / "prepare_report.json").write_text(json.dumps(report, indent=1))

    print(f"""
  Done -> {out_root}

    images                 {len(stems)}   (train {counts['train']} / valid {counts['valid']} / test {counts['test']})
    button instances       {kept}
    OCR-scorable           {scorable}
    symbol / unreadable    {kept - scorable}
    dropped as non-button  {total_in - kept}
    duplicate groups held together {len(dup_groups)}

    data.yaml              {out_root}/data.yaml        (nc=1, names=['button'])
    OCR ground truth       {out_root}/ocr_ground_truth.json
    stats                  {out_root}/prepare_report.json
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
