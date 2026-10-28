"""Annotated output images."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

GREEN = (60, 220, 60)      # read
AMBER = (0, 180, 255)      # inferred from the grid
GREY = (150, 150, 150)     # detected but unreadable
MAGENTA = (220, 60, 220)   # the selected target


def _label(img, text, x, y, colour, scale=0.5, thick=1):
    """Draw text on a filled background; (x, y) is the baseline-left corner."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    h, w = img.shape[:2]
    x = max(0, min(x, w - tw - 5))
    y = max(th + 3, min(y, h - 3))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), colour, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (10, 10, 10), thick, cv2.LINE_AA)


def annotate(image: np.ndarray, dets: list[Detection],
             selected: Detection | None = None,
             show_confidence: bool = True) -> np.ndarray:
    out = image.copy()
    # Scale everything with the image so 4000px phone photos stay readable.
    s = max(1.0, min(out.shape[:2]) / 640.0)
    thick = max(1, int(round(2 * s)))
    scale = 0.5 * s

    for d in dets:
        x1, y1, x2, y2 = d.bbox
        colour = GREY if not d.text else (AMBER if d.inferred else GREEN)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)

        txt = d.text or "?"
        # Only append the confidence if the box is wide enough to hold it.
        # Dense panels have ~35px buttons and the extra 5 characters turn the
        # whole thing into overlapping soup.
        if show_confidence and d.text and d.width > 55 * s:
            txt += f" {d.confidence * d.ocr_confidence:.2f}"
        if d.inferred:
            txt += "*"
        # Inside the box, not above it - above meant every label overlapped the
        # button above and the whole panel read as if labels were off by one.
        _label(out, txt, x1 + 2, y1 + int(14 * s), colour, scale, max(1, thick // 2))

    if selected is not None:
        x1, y1, x2, y2 = selected.bbox
        pad = int(4 * s)
        cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                      MAGENTA, thick + 1)
        cv2.drawMarker(out, selected.center, MAGENTA, cv2.MARKER_CROSS,
                       int(18 * s), thick + 1)
        _label(out, f"TARGET {selected.text}", x1 - pad, y2 + int(20 * s),
               MAGENTA, scale * 1.1, max(1, thick // 2))

    return out


def legend(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    s = max(1.0, min(out.shape[:2]) / 640.0)
    x, y = int(8 * s), int(16 * s)
    for name, colour in [("read", GREEN), ("inferred*", AMBER),
                         ("unread", GREY), ("target", MAGENTA)]:
        _label(out, name, x, y, colour, 0.45 * s, 1)
        x += int((len(name) + 3) * 9 * s)
    return out
