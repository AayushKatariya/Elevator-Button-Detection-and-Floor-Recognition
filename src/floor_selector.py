"""
Given every detected button and a target floor, decide which one to press.

Two pieces of work happen here. Grid inference fills in buttons the recognizer
couldn't read, using the fact that floor numbers run in order down a column.
The matching cascade then picks a button for the query, or refuses.

Refusing matters. Recognition tops out around 66%, so the interesting question
is not "can we always answer" but "when we answer, are we right". Pressing the
wrong floor is worse than admitting we don't know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .detector import Detection
from .ocr import normalize


@dataclass
class Selection:
    target_floor: str
    button: Detection | None
    reason: str = "ok"          # ok | not_detected | ocr_failed | no_match
    match_type: str = "exact"   # exact | fuzzy | inferred | none
    ambiguous: bool = False

    def to_dict(self, confidence_mode: str = "combined") -> dict:
        if self.button is None:
            return {"target_floor": _pretty(self.target_floor),
                    "button": None, "reason": self.reason}

        b = self.button
        conf = (b.confidence * b.ocr_confidence if confidence_mode == "combined"
                else b.confidence)
        out = {
            "target_floor": _pretty(self.target_floor),
            "button": {
                "text": b.text,
                "bbox": list(b.bbox),
                "center": list(b.center),
                "confidence": round(float(conf), 4),
            },
        }
        # Extra keys only when something is off the happy path, so a normal
        # result matches the schema in the assignment exactly.
        if self.match_type != "exact":
            out["button"]["match_type"] = self.match_type
        if b.inferred:
            out["button"]["inferred"] = True
        if self.ambiguous:
            out["button"]["ambiguous"] = True
        return out


def _pretty(target: str):
    """Numeric floors come back as ints so the JSON reads `27`, not `"27"`."""
    return int(target) if re.fullmatch(r"-?\d+", target) else target


def normalize_target(target) -> str:
    """
    Push the query through the same normalization as the OCR output.

    Comparing a raw target against normalized readings is an easy way to get
    silent misses ('27 ' or 'b1' would never match), so both sides go through
    the same function.
    """
    return normalize(str(target))


def levenshtein(a: str, b: str, cap: int = 2) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --- grid inference -------------------------------------------------------

def _cluster(values: list[float], tol: float) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    idx = [0] * len(values)
    c = 0
    for n, i in enumerate(order):
        if n and values[i] - values[order[n - 1]] > tol:
            c += 1
        idx[i] = c
    return idx


def infer_missing_labels(dets: list[Detection], enabled: bool = True) -> int:
    """
    Fill unread buttons in from their neighbours.

    Column by column: if the nearest read buttons above and below a gap carry
    consecutive numbers and the spacing is even, the gap is the number in
    between. Most unread buttons are unread because of glare, not because
    they're missing, so this recovers a fair few.

    Anything filled in gets inferred=True and a low confidence. It is never
    passed off as something the recognizer actually read.
    """
    if not enabled or len(dets) < 3:
        return 0

    heights = sorted(d.height for d in dets)
    widths = sorted(d.width for d in dets)
    med_h = heights[len(heights) // 2] or 1
    med_w = widths[len(widths) // 2] or 1

    by_col: dict[int, list[Detection]] = {}
    for d, c in zip(dets, _cluster([d.center[0] for d in dets], med_w * 0.6)):
        by_col.setdefault(c, []).append(d)

    filled = 0
    for column in by_col.values():
        if len(column) < 3:
            continue
        column.sort(key=lambda d: d.center[1])

        for i, d in enumerate(column):
            if d.text:
                continue

            above = below = None
            for j in range(i - 1, -1, -1):
                if column[j].text and column[j].text.lstrip("-").isdigit():
                    above = (j, int(column[j].text))
                    break
            for j in range(i + 1, len(column)):
                if column[j].text and column[j].text.lstrip("-").isdigit():
                    below = (j, int(column[j].text))
                    break
            if not (above and below):
                continue

            # Higher floors sit higher up, so the number above should be larger,
            # and the drop should match the number of slots between them.
            slots = below[0] - above[0]
            if slots <= 0 or above[1] - below[1] != slots:
                continue
            if abs(d.center[1] - column[above[0]].center[1]) > med_h * 2.5 * (i - above[0]):
                continue

            d.text = str(above[1] - (i - above[0]))
            d.inferred = True
            d.ocr_confidence = 0.35
            d.meta["inferred_from"] = [column[above[0]].text, column[below[0]].text]
            filled += 1

    return filled


# --- selection ------------------------------------------------------------

def select_button(dets: list[Detection], target, *, fuzzy_max_distance: int = 1,
                  enable_grid_inference: bool = True,
                  confidence_mode: str = "combined",
                  vocabulary: set[str] | None = None) -> Selection:
    """
    Pick the button for `target`, or say why we can't.

    Passing `vocabulary` (the known label strings) makes fuzzy matching much
    safer -- see below.
    """
    want = normalize_target(target)

    if not dets:
        return Selection(want, None, reason="not_detected", match_type="none")

    infer_missing_labels(dets, enable_grid_inference)

    readable = [d for d in dets if d.text]
    if not readable:
        return Selection(want, None, reason="ocr_failed", match_type="none")

    def score(d: Detection) -> float:
        return (d.confidence * d.ocr_confidence if confidence_mode == "combined"
                else d.confidence)

    exact = [d for d in readable if d.text == want]
    if exact:
        exact.sort(key=score, reverse=True)
        best = exact[0]
        return Selection(want, best,
                         match_type="inferred" if best.inferred else "exact",
                         ambiguous=len(exact) > 1)

    # Fuzzy matching, with two guards. Both were added after it produced
    # confidently wrong floors:
    #
    #   Same length only. Levenshtein('20', '2') is 1, so plain edit distance
    #   answered a query for floor 20 with the button labelled 2.
    #
    #   Never offer a candidate that is itself a known label. A button reading
    #   '22' means floor 22; it is not a misreading of '20'. Without this, '20'
    #   matched '22' and 'B1' matched '21'. Fuzzy matching is for rescuing
    #   garbage readings, not for swapping one real floor for another.
    if fuzzy_max_distance > 0:
        cands = [d for d in readable if len(d.text) == len(want)]
        if vocabulary:
            cands = [d for d in cands if d.text not in vocabulary]
        near = [(levenshtein(d.text, want, fuzzy_max_distance), d) for d in cands]
        near = [(dist, d) for dist, d in near if dist <= fuzzy_max_distance]
        if near:
            near.sort(key=lambda t: (t[0], -score(t[1])))
            closest = near[0][0]
            return Selection(want, near[0][1], match_type="fuzzy",
                             ambiguous=sum(1 for d, _ in near if d == closest) > 1)

    return Selection(want, None, reason="no_match", match_type="none")
