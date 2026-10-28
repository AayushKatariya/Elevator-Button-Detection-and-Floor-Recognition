# Evaluation Report

Elevator Button Detection & Floor Recognition - Lastmile Robotics Assignment 05

- Detector: YOLOv8n (single class `button`), `models/best.pt`
- Recognizer: `hybrid` (YOLOv8n-cls primary, PaddleOCR fallback)
- Split: held-out **test** (Final dataset/test)
- Hardware: NVIDIA GeForce RTX 5070 Ti

## Detection

```
Precision:         0.9111
Recall:            0.9063
mAP@50:            0.9489
mAP@50:95:         0.6804
Inference latency: 7.46 ms  (134 FPS), batch=1
```

Latency measured over 120 images at batch=1 after 10 discarded warm-up runs; mean / median / p95 = 7.46 / 6.66 / 9.05 ms.

Full pipeline (detection + OCR) is **1768.9 ms** (0.6 FPS) - recognition dominates, not detection.

## OCR

```
Number of buttons evaluated: 1323
Correctly recognized:        877
Incorrectly recognized:      446
OCR accuracy:                66.3%
```

**Denominator.** Denominator = ground-truth buttons that were DETECTED (IoU>=0.5) and carry readable text. Undetected buttons appear in the end-to-end number instead; symbol/unreadable buttons are excluded entirely.

- Readable buttons in the split: 1385
- Not detected (counted in end-to-end instead): 62
- Symbol / unreadable, excluded: 661
- Returned no read at all: 3

### Accuracy by button size

| Button size | Correct | Total | Accuracy |
|---|---|---|---|
| tiny(<24) | 40 | 169 | 23.7% |
| small(24-32) | 92 | 150 | 61.3% |
| med(32-48) | 323 | 467 | 69.2% |
| large(48-64) | 162 | 211 | 76.8% |
| xl(>=64) | 260 | 326 | 79.8% |

Button size is the dominant factor in recognition accuracy.

### Most common misreads

| Truth | Read as | Count |
|---|---|---|
| `7` | `1` | 10 |
| `3` | `2` | 10 |
| `3` | `1` | 7 |
| `2` | `7` | 5 |
| `11` | `17` | 5 |
| `6` | `B` | 4 |
| `3` | `5` | 4 |
| `6` | `G` | 4 |
| `5` | `6` | 4 |
| `2` | `1` | 4 |
| `8` | `9` | 4 |
| `9` | `5` | 4 |

## End-to-End

```
Correct floor identification: 861
Total test cases:             1385
Accuracy:                     62.2%
```

One test case per readable ground-truth button: query that floor, then check the returned box against the correct button at IoU >= 0.5.

### Failure modes

| Failure mode | Count |
|---|---|
| detected, OCR misread | 418 |
| button not detected | 59 |
| wrong button selected | 44 |
| detected, OCR failed (no read) | 3 |

## Training configuration

```
model:      yolov8n.pt
epochs:     100 / 100
imgsz:      640
batch:      16
seed:       42
time:       0h 13m 34s
hardware:   NVIDIA GeForce RTX 5070 Ti
```
