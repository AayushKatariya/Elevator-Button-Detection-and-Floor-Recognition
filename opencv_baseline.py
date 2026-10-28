#!/usr/bin/env python3
"""
Part 3 option: "OpenCV-based methods".

Three classical recognizers, all built with cv2 primitives only (no torch, no
sklearn), evaluated on exactly the same test crops as the OCR engines so the
comparison is like-for-like:

  1. Template matching  -- per-class mean image, ranked by normalised
                           cross-correlation. The most classical approach there is.
  2. kNN on pixels      -- cv2.ml.KNearest over contrast-normalised crops.
  3. HOG + SVM          -- cv2.HOGDescriptor features into cv2.ml.SVM (RBF).
                           The standard pre-deep-learning recipe for digits.

All three are closed-set: they choose among label strings seen in training and
cannot read a novel label. On this test split that costs at most 0.8%.

Usage:
    python opencv_baseline.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Set from the crops actually on disk. export_crops.py takes a --size flag, and
# hardcoding this silently broke the script when the crops were re-exported at
# 128px instead of 64.
SIZE = 64


def load(path="crops/arrays.npz"):
    d = np.load(path, allow_pickle=True)
    global SIZE
    SIZE = int(d["train_X"].shape[1])
    return ((d["train_X"], d["train_y"]), (d["valid_X"], d["valid_y"]),
            (d["test_X"], d["test_y"]))


def norm_img(x: np.ndarray) -> np.ndarray:
    """Per-crop contrast normalisation - the panels vary wildly in exposure."""
    x = cv2.equalizeHist(x)
    return x.astype(np.float32) / 255.0


def hog_desc():
    return cv2.HOGDescriptor(
        _winSize=(SIZE, SIZE), _blockSize=(16, 16), _blockStride=(8, 8),
        _cellSize=(8, 8), _nbins=9)


def evaluate(name, predict, Xte, yte, extra=None):
    t0 = time.perf_counter()
    pred = predict(Xte)
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(Xte))
    correct = int(sum(1 for p, t in zip(pred, yte) if p == t))
    by_len = defaultdict(lambda: [0, 0])
    conf = Counter()
    for p, t in zip(pred, yte):
        by_len[len(t)][1] += 1
        if p == t:
            by_len[len(t)][0] += 1
        else:
            conf[(t, p)] += 1
    acc = correct / max(1, len(yte))
    print(f"  {name:<22}{100*acc:6.1f}%   {ms:7.2f} ms/crop")
    return {"method": name, "accuracy": round(acc, 4), "correct": correct,
            "evaluated": len(yte), "ms_per_crop": round(ms, 3),
            "by_label_len": {str(k): {"correct": v[0], "total": v[1]}
                             for k, v in sorted(by_len.items())},
            "top_confusions": [{"truth": t, "got": p, "n": c}
                               for (t, p), c in conf.most_common(15)],
            **(extra or {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", default="crops/arrays.npz")
    ap.add_argument("--out", default="reports/opencv_baseline.json")
    args = ap.parse_args()

    (Xtr, ytr), (Xva, yva), (Xte, yte) = load(args.crops)
    # train on train+valid: the OCR engines were never "trained" at all, so
    # giving the classical methods every labelled crop that is not test is fair.
    Xtr = np.concatenate([Xtr, Xva]); ytr = np.concatenate([ytr, yva])
    labels = sorted(set(ytr.tolist()))
    idx = {l: i for i, l in enumerate(labels)}
    print(f"OpenCV classical baselines\n  train {len(Xtr)}  test {len(Xte)}  "
          f"classes {len(labels)}  crop {SIZE}x{SIZE}\n")
    print(f"  {'method':<22}{'accuracy':>7}   {'speed':>7}")
    print("  " + "-" * 42)
    results = []

    # ---- 1. template matching -------------------------------------------
    tmpl = np.zeros((len(labels), SIZE, SIZE), np.float32)
    cnt = np.zeros(len(labels), np.float32)
    for x, y in zip(Xtr, ytr):
        tmpl[idx[y]] += norm_img(x); cnt[idx[y]] += 1
    tmpl /= np.maximum(cnt, 1)[:, None, None]
    T = tmpl.reshape(len(labels), -1)
    T = (T - T.mean(1, keepdims=True))
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)

    def pred_tmpl(X):
        V = np.stack([norm_img(x).ravel() for x in X])
        V -= V.mean(1, keepdims=True)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-6)
        return [labels[i] for i in (V @ T.T).argmax(1)]

    results.append(evaluate("template matching", pred_tmpl, Xte, yte))

    # ---- 2. kNN on pixels ------------------------------------------------
    small = 20  # kNN on 64x64 is slow and no more accurate
    def feats(X):
        return np.stack([cv2.resize(norm_img(x), (small, small)).ravel() for x in X])

    knn = cv2.ml.KNearest_create()
    knn.train(feats(Xtr), cv2.ml.ROW_SAMPLE,
              np.array([idx[y] for y in ytr], np.int32))

    def pred_knn(X, k=3):
        _, r, _, _ = knn.findNearest(feats(X), k)
        return [labels[int(i)] for i in r.ravel()]

    results.append(evaluate("kNN (k=3, pixels)", pred_knn, Xte, yte))

    # ---- 3. HOG + SVM ----------------------------------------------------
    hog = hog_desc()
    def hogf(X):
        return np.stack([hog.compute(cv2.equalizeHist(x)).ravel() for x in X])

    Ftr = hogf(Xtr)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(12.5); svm.setGamma(0.5 / max(1, Ftr.shape[1]))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 1e-4))
    print("  (training SVM on HOG features, this takes a minute ...)")
    svm.train(Ftr, cv2.ml.ROW_SAMPLE, np.array([idx[y] for y in ytr], np.int32))

    def pred_svm(X):
        return [labels[int(i)] for i in svm.predict(hogf(X))[1].ravel()]

    results.append(evaluate("HOG + SVM (RBF)", pred_svm, Xte, yte,
                            extra={"hog_dim": int(Ftr.shape[1])}))

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  best classical method: {best['method']} at {100*best['accuracy']:.1f}%")
    print("\n  accuracy by label length (best method):")
    for k, v in best["by_label_len"].items():
        if v["total"]:
            print(f"    {k}-char: {v['correct']:5d}/{v['total']:<5d} {100*v['correct']/v['total']:5.1f}%")

    Path("reports").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"train_size": int(len(Xtr)), "test_size": int(len(Xte)),
         "classes": len(labels), "crop_size": SIZE, "results": results},
        indent=1), encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
