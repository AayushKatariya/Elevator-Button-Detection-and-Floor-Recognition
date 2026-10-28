"""
Detector + recognizer + floor selector, wired together.

inference.py, detect_floor.py and evaluate.py all go through this so they can't
drift apart. Behaviour changes belong in config.yaml.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .detector import ButtonDetector, Detection
from .floor_selector import Selection, select_button
from .ocr import ButtonClassifier, build_engine, crop_button


def load_config(path: str | Path = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


class ElevatorPipeline:
    def __init__(self, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        d = self.cfg["detection"]
        self.detector = ButtonDetector(
            weights=self.cfg["paths"]["model_weights"],
            conf=d["conf_threshold"], iou=d["iou_threshold"],
            max_det=d["max_detections"],
            imgsz=self.cfg["training"]["imgsz"], device=device)
        self.engine = build_engine(self.cfg)
        self.padding = self.cfg["ocr"]["crop_padding_ratio"]
        self._timings: dict[str, float] = {}
        self._vocab: set[str] | None = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        dets = self.detector.detect(image)
        self._timings["detect_ms"] = (time.perf_counter() - t0) * 1000
        return dets

    def read(self, image: np.ndarray, dets: list[Detection]) -> list[Detection]:
        """OCR each detection on its own crop, never on the full panel."""
        t0 = time.perf_counter()
        crops = [crop_button(image, d.bbox, self.padding) for d in dets]

        if isinstance(self.engine, ButtonClassifier):
            # Batching is much faster than one call per button.
            for d, r in zip(dets, self.engine.read_batch(crops)):
                d.text, d.ocr_confidence = r.text, r.confidence
        else:
            for d, crop in zip(dets, crops):
                r = self.engine.read_crop(crop)
                d.text, d.ocr_confidence = r.text, r.confidence
                if r.variant:
                    d.meta["engine"] = r.variant

        self._timings["ocr_ms"] = (time.perf_counter() - t0) * 1000
        return dets

    def run(self, image: np.ndarray) -> list[Detection]:
        return self.read(image, self.detect(image))

    @property
    def vocabulary(self) -> set[str]:
        """Known label strings; keeps fuzzy matching from inventing floors."""
        if self._vocab is None:
            clf = getattr(self.engine, "classifier", self.engine)
            self._vocab = getattr(clf, "vocabulary", set())
        return self._vocab

    def find_floor(self, image: np.ndarray, target) -> tuple[Selection, list[Detection]]:
        dets = self.run(image)
        fs = self.cfg["floor_selection"]
        sel = select_button(dets, target,
                            fuzzy_max_distance=fs["fuzzy_match_max_distance"],
                            enable_grid_inference=fs["enable_grid_inference"],
                            confidence_mode=fs["confidence_mode"],
                            vocabulary=self.vocabulary)
        return sel, dets

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

    @staticmethod
    def imread(path: str | Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return img

    def result_json(self, dets: list[Detection], image_path: str | None = None) -> dict:
        combined = self.cfg["floor_selection"]["confidence_mode"] == "combined"
        buttons = []
        for d in dets:
            item = d.to_dict()
            if combined:
                item["confidence"] = round(d.confidence * d.ocr_confidence, 4)
                item["detection_confidence"] = round(d.confidence, 4)
            buttons.append(item)

        out = {"buttons": buttons, "count": len(buttons)}
        if image_path:
            out["image"] = str(image_path)
        out["timings_ms"] = {k: round(v, 2) for k, v in self._timings.items()}
        return out


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
