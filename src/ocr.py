"""
Reading the text off a button crop.

OCR always runs on the crop, never the whole panel. Everything here takes one
crop and returns one short string.

Five recognizers live here because Part 3 of the assignment asks for a
justified choice and the only honest way to justify one is to measure them all.
`ocr_compare.py` does that; the winner is the hybrid at the bottom of this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# The 34 characters that actually appear in the 258 ground-truth labels.
DEFAULT_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPRSTUVXZ-"


@dataclass
class OcrResult:
    text: str | None
    confidence: float
    raw: str = ""        # engine output before normalization
    variant: str = ""    # which preprocessing/engine produced it


# --- crops ----------------------------------------------------------------

def crop_button(image: np.ndarray, xyxy, padding: float = 0.15) -> np.ndarray:
    """Crop a box, expanded (or shrunk, if padding is negative) on all sides."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    x1 = max(0, int(round(x1 - pw)))
    y1 = max(0, int(round(y1 - ph)))
    x2 = min(w, int(round(x2 + pw)))
    y2 = min(h, int(round(y2 + ph)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), np.uint8)
    return image[y1:y2, x1:x2]


def _upscale(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0 or h >= target_h:
        return img
    s = target_h / h
    return cv2.resize(img, (max(1, int(round(w * s))), target_h),
                      interpolation=cv2.INTER_LINEAR)


def preprocess_variants(crop: np.ndarray, target_h: int = 128) -> list[tuple[str, np.ndarray]]:
    """
    Three renderings of the same crop, for test-time augmentation.

    Panels carry both dark-on-light and light-on-dark buttons and the glare
    varies wildly, so no single preprocessing wins everywhere. Trying all three
    and keeping the most confident read is cheap: worth ~5 points for EasyOCR,
    ~1 for PaddleOCR.
    """
    if crop.size == 0:
        return []
    big = _upscale(crop, target_h)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return [("gray", gray), ("clahe", clahe), ("invert", cv2.bitwise_not(clahe))]


# --- normalization --------------------------------------------------------

# Only characters that are never a label in their own right. 'L' is absent on
# purpose: L is a real label and so are L1-L7, so mapping L->1 turns 'L1' into
# '11'. Same goes for B, G, S, D, T, Z, which all head real label families.
_ALWAYS = {"Q": "0", "|": "1", "!": "1"}

# There is deliberately no general letter->digit coercion. An earlier version
# had one and it was worse than useless: normalize('B1') came back '81' and
# 'L1' came back '11'. Since select_button() normalizes the query too, that
# silently broke every basement and lobby lookup. Glyph confusion is handled by
# snap_to_vocabulary() instead, which can only ever map onto a real label.


def normalize(raw: str, allowlist: str = DEFAULT_ALLOWLIST) -> str:
    if not raw:
        return ""
    s = re.sub(r"[\s._,:;'\"()\[\]]+", "", raw.strip().upper())
    s = "".join(_ALWAYS.get(ch, ch) for ch in s)
    s = "".join(ch for ch in s if ch in allowlist)
    return {"GF": "G", "BF": "B", "1F": "1"}.get(s, s)


def snap_to_vocabulary(s: str, vocab: set[str], max_distance: int = 1) -> tuple[str, bool]:
    """
    Pull a reading onto the nearest known label, if one is close enough.

    Returns (text, was_snapped) so callers can flag inexact matches rather than
    passing them off as direct reads.
    """
    if not s or s in vocab:
        return s, False

    def dist(a: str, b: str) -> int:
        if abs(len(a) - len(b)) > max_distance:
            return max_distance + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    best, bestd = None, max_distance + 1
    for cand in vocab:
        d = dist(s, cand)
        if d < bestd:
            best, bestd = cand, d
    return (best, True) if best and bestd <= max_distance else (s, False)


# --- engines --------------------------------------------------------------

class EasyOcrEngine:
    def __init__(self, allowlist: str = DEFAULT_ALLOWLIST, min_confidence: float = 0.30,
                 target_crop_height: int = 128, gpu: bool = True):
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self.target_crop_height = target_crop_height
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        return self._reader

    def read_crop(self, crop: np.ndarray, use_tta: bool = True) -> OcrResult:
        variants = preprocess_variants(crop, self.target_crop_height)
        if not variants:
            return OcrResult(None, 0.0)
        if not use_tta:
            variants = variants[:1]

        best = OcrResult(None, 0.0)
        for name, img in variants:
            # recognize(), not readtext(). readtext() runs CRAFT text detection
            # first, which finds nothing in a ~30px single-character crop -- it
            # returned no read at all for 71% of them. We already know where the
            # text is; it's the whole crop.
            try:
                out = self.reader.recognize(img, horizontal_list=None, free_list=None,
                                            allowlist=self.allowlist, detail=1)
            except Exception:
                continue
            for item in out:
                text, conf = item[1], item[2]
                norm = normalize(text, self.allowlist)
                if norm and conf > best.confidence:
                    best = OcrResult(norm, float(conf), raw=text, variant=name)

        if best.text and best.confidence < self.min_confidence:
            return OcrResult(None, 0.0, raw=best.raw, variant=best.variant)
        return best


class TesseractEngine:
    """
    Baseline for the Part 3 comparison. Needs the Tesseract system binary, not
    just pytesseract (`winget install UB-Mannheim.TesseractOCR` on Windows).

    Page-segmentation mode matters more than anything else here: --psm 10
    ("single character") and --psm 8 ("single word") are the only ones that fit
    an isolated button. Otsu binarization is added to the variants since
    Tesseract wants clean black-on-white, which engraved metal rarely gives it.
    """

    def __init__(self, allowlist: str = DEFAULT_ALLOWLIST, min_confidence: float = 0.30,
                 target_crop_height: int = 128, binary_path: str | None = None):
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self.target_crop_height = target_crop_height
        import pytesseract
        if binary_path:
            pytesseract.pytesseract.tesseract_cmd = binary_path
        self._pt = pytesseract

    def read_crop(self, crop: np.ndarray, use_tta: bool = True) -> OcrResult:
        variants = preprocess_variants(crop, self.target_crop_height)
        if not variants:
            return OcrResult(None, 0.0)
        if use_tta:
            _, otsu = cv2.threshold(variants[0][1], 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants += [("otsu", otsu), ("otsu_inv", cv2.bitwise_not(otsu))]
        else:
            variants = variants[:1]

        best = OcrResult(None, 0.0)
        for name, img in variants:
            for psm in (10, 8):
                cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist={self.allowlist}"
                try:
                    data = self._pt.image_to_data(img, config=cfg,
                                                  output_type=self._pt.Output.DICT)
                except Exception:
                    continue
                for text, conf in zip(data.get("text", []), data.get("conf", [])):
                    try:
                        c = float(conf) / 100.0
                    except (TypeError, ValueError):
                        continue
                    if c < 0:
                        continue
                    norm = normalize(text, self.allowlist)
                    if norm and c > best.confidence:
                        best = OcrResult(norm, c, raw=text, variant=f"{name}/psm{psm}")

        if best.text and best.confidence < self.min_confidence:
            return OcrResult(None, 0.0, raw=best.raw, variant=best.variant)
        return best


class PaddleOcrEngine:
    """
    The strongest of the three general-purpose engines here, and the fallback
    half of the hybrid.

    Uses TextRecognition directly rather than the full PaddleOCR pipeline. Two
    reasons: the detection stage is pointless when the text fills the crop, and
    on this machine it crashes outright with a oneDNN NotImplementedError.
    """

    def __init__(self, allowlist: str = DEFAULT_ALLOWLIST, min_confidence: float = 0.30,
                 target_crop_height: int = 128, gpu: bool = False):
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self.target_crop_height = target_crop_height
        self.gpu = gpu
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            from paddleocr import TextRecognition
            self._ocr = TextRecognition()
        return self._ocr

    def _recognize(self, img: np.ndarray) -> list[tuple[str, float]]:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        try:
            out = list(self.ocr.predict(rgb))
        except Exception:
            return []
        res = []
        for item in out:
            if isinstance(item, dict) and "rec_text" in item:
                try:
                    res.append((str(item["rec_text"]), float(item.get("rec_score", 0.0))))
                except (TypeError, ValueError):
                    pass
        return res

    def read_crop(self, crop: np.ndarray, use_tta: bool = True) -> OcrResult:
        variants = preprocess_variants(crop, self.target_crop_height)
        if not variants:
            return OcrResult(None, 0.0)
        if not use_tta:
            variants = variants[:1]

        best = OcrResult(None, 0.0)
        for name, img in variants:
            for text, conf in self._recognize(img):
                norm = normalize(text, self.allowlist)
                if norm and conf > best.confidence:
                    best = OcrResult(norm, float(conf), raw=text, variant=name)

        if best.text and best.confidence < self.min_confidence:
            return OcrResult(None, 0.0, raw=best.raw, variant=best.variant)
        return best


# --- classifier + hybrid --------------------------------------------------

def _unsafe_name(dirname: str) -> str:
    """Inverse of export_crops.safe_name(): 'cls_neg1' -> '-1'."""
    return re.sub(r"^cls_", "", dirname).replace("neg", "-")


class ButtonClassifier:
    """
    A ~1.8M parameter YOLOv8n-cls over button crops.

    Picks one of the 228 labels it saw in training instead of decoding
    characters. That is a much easier question than open-ended OCR, which is
    why it beats every general engine measured here (66.9% against PaddleOCR's
    59.4%) while running 33x faster.

    The catch is that it has no output slot for a label it never saw. That's
    11 of 1385 test buttons, but it would be worse on an unfamiliar panel --
    hence HybridEngine.
    """

    def __init__(self, weights: str = "models/button_cls.pt", device: str | None = None):
        self.weights = weights
        self.device = device
        self._model = None
        self._names: dict[int, str] = {}

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            self._names = {int(k): _unsafe_name(v) for k, v in self._model.names.items()}
        return self._model

    @property
    def vocabulary(self) -> set[str]:
        _ = self.model
        return set(self._names.values())

    def read_crop(self, crop: np.ndarray, use_tta: bool = False) -> OcrResult:
        if crop is None or crop.size == 0:
            return OcrResult(None, 0.0)
        try:
            r = self.model.predict(crop, verbose=False, device=self.device)[0]
        except Exception:
            return OcrResult(None, 0.0)
        idx = int(r.probs.top1)
        name = self._names.get(idx)
        return OcrResult(name, float(r.probs.top1conf), raw=name or "", variant="cls")

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult]:
        out = [OcrResult(None, 0.0) for _ in crops]
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size]
        if not valid:
            return out
        try:
            preds = self.model.predict([c for _, c in valid], verbose=False,
                                       device=self.device)
        except Exception:
            return out
        for (i, _), r in zip(valid, preds):
            idx = int(r.probs.top1)
            name = self._names.get(idx)
            out[i] = OcrResult(name, float(r.probs.top1conf), raw=name or "", variant="cls")
        return out


class HybridEngine:
    """
    Classifier first, PaddleOCR when the classifier is unsure.

    Each half covers the other's blind spot: the classifier is accurate and
    fast but closed-set, PaddleOCR reads arbitrary strings but was never
    trained on engraved metal or specular glare.
    """

    def __init__(self, classifier: ButtonClassifier, fallback=None,
                 classifier_threshold: float = 0.55, min_confidence: float = 0.30):
        self.classifier = classifier
        self.fallback = fallback
        self.classifier_threshold = classifier_threshold
        self.min_confidence = min_confidence

    def read_crop(self, crop: np.ndarray, use_tta: bool = True) -> OcrResult:
        primary = self.classifier.read_crop(crop)
        if primary.text and primary.confidence >= self.classifier_threshold:
            return primary

        if self.fallback is None:
            good = primary.text and primary.confidence >= self.min_confidence
            return primary if good else OcrResult(None, 0.0, raw=primary.raw)

        alt = self.fallback.read_crop(crop, use_tta=use_tta)
        # Ties go to the classifier, which is the stronger model on average.
        if alt.text and alt.confidence > primary.confidence:
            return OcrResult(alt.text, alt.confidence, raw=alt.raw, variant="paddle")
        if primary.text and primary.confidence >= self.min_confidence:
            return OcrResult(primary.text, primary.confidence,
                             raw=primary.raw, variant="cls_low")
        if alt.text and alt.confidence >= self.min_confidence:
            return OcrResult(alt.text, alt.confidence, raw=alt.raw, variant="paddle")
        return OcrResult(None, 0.0, raw=primary.raw or alt.raw, variant="none")


def build_engine(cfg: dict):
    """Construct whichever engine config.yaml names."""
    o = cfg["ocr"]
    name = o.get("engine", "hybrid")
    common = dict(allowlist=o["allowlist"], min_confidence=o["min_confidence"],
                  target_crop_height=o["target_crop_height"])

    if name == "easyocr":
        return EasyOcrEngine(**common)
    if name == "tesseract":
        return TesseractEngine(**common, binary_path=o.get("tesseract_path"))
    if name == "paddleocr":
        return PaddleOcrEngine(**common)
    if name == "classifier":
        return ButtonClassifier(weights=cfg["paths"]["classifier_weights"])
    if name == "hybrid":
        fallback = PaddleOcrEngine(**common) if o.get("hybrid_fallback", True) else None
        return HybridEngine(
            ButtonClassifier(weights=cfg["paths"]["classifier_weights"]),
            fallback=fallback,
            classifier_threshold=o.get("classifier_threshold", 0.55),
            min_confidence=o["min_confidence"])

    raise ValueError(f"unknown ocr.engine: {name!r}")
