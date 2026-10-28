"""YOLOv8n button detector. Single class - the label comes from the OCR stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """One button, in original-image pixel coordinates."""
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    confidence: float
    text: str | None = None
    ocr_confidence: float = 0.0
    inferred: bool = False              # text came from grid inference, not a read
    meta: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "confidence": round(self.confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
        }
        if self.inferred:
            d["inferred"] = True
        return d


class ButtonDetector:
    def __init__(self, weights: str = "models/best.pt", conf: float = 0.25,
                 iou: float = 0.45, max_det: int = 100, imgsz: int = 640,
                 device: str | None = None):
        self.weights = weights
        self.conf, self.iou, self.max_det, self.imgsz = conf, iou, max_det, imgsz
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not Path(self.weights).exists():
                raise FileNotFoundError(
                    f"{self.weights} not found - run: python train.py")
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Find buttons in a BGR image.

        Ultralytics letterboxes internally but returns boxes in original image
        coordinates, so a 4032x3024 phone photo and a 640x640 dataset image both
        come back in their own pixel space. tests/test_coordinate_space.py
        checks this, because it would be an easy thing to get silently wrong and
        every image in the dataset is square (no letterbox padding, so the bug
        would never show up in evaluation).
        """
        res = self.model.predict(image, conf=self.conf, iou=self.iou,
                                 max_det=self.max_det, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None:
            return []

        H, W = image.shape[:2]
        out = []
        for box, score in zip(res.boxes.xyxy.cpu().numpy(),
                              res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
            if x2 > x1 and y2 > y1:
                out.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(score)))

        # Reading order, so visualizations and grid inference are deterministic.
        out.sort(key=lambda d: (d.center[1], d.center[0]))
        return out
