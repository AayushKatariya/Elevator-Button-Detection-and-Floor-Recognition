# Elevator Button Detection & Floor Recognition

Lastmile Robotics — Assignment 05.

Given a photograph of an elevator control panel and a target floor, this pipeline
detects every button, reads the text on each one, and returns the pixel
coordinates of the button to press.

```bash
python detect_floor.py --image examples/input/1957_jpg.jpg --floor 12
```

```json
{
  "target_floor": 12,
  "button": {
    "text": "12",
    "bbox": [370, 346, 435, 397],
    "center": [402, 371],
    "confidence": 0.86
  }
}
```

`stdout` carries only that JSON — all logging goes to stderr — so it pipes
directly into another program.

---

## Contents

- [Quick start](#quick-start)
- [Dataset selection](#dataset-selection)
- [Dataset used](#dataset-used)
- [Dataset preparation](#dataset-preparation)
- [Model architecture](#model-architecture)
- [Training configuration](#training-configuration)
- [Hyperparameters](#hyperparameters)
- [Evaluation results](#evaluation-results)
- [OCR approach](#ocr-approach)
- [Floor-selection logic](#floor-selection-logic)
- [Inference instructions](#inference-instructions)
- [Known limitations](#known-limitations)
- [What I would improve with more time](#what-i-would-improve-with-more-time)
- [Repository layout](#repository-layout)

---

## Quick start

```bash
# 1. PyTorch first — see the GPU note in requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Build the merged dataset from the two raw exports (~3 min)
python prepare_dataset.py

# 3. Train the detector (~14 min on an RTX 5070 Ti)
python train.py

# 4. Train the recognizer
python export_crops.py --size 128
python train_classifier.py --imgsz 128 --epochs 120

# 5. Use it
python detect_floor.py --image <panel.jpg> --floor 12
python inference.py --image <panel.jpg> --visualize
python evaluate.py
```

Pre-trained weights are included in `models/`, so steps 2–4 are only needed to
reproduce training.

---

## Dataset selection

Four candidate sources were evaluated before choosing:

| Dataset | Images | Per-button text labels? | Verdict |
|---|---|---|---|
| **Sun Moon University / Zhu et al.** | 2,019 | **Yes — 368 classes ARE the button text** | **Chosen** |
| andrey-rel (segmentation re-export) | 2,019 | Corrupt dictionary | Used for its sharper images only |
| CUHK large-scale ([arXiv:2103.09030](https://arxiv.org/abs/2103.09030)) | 3,718 | Yes | **Unobtainable** — the only download link returns HTTP 404 (verified Aug 2026), no mirror exists |
| Typical Roboflow elevator sets | 400–800 | No — a single `button` class | Rejected: no OCR ground truth |

What decided it was per-button text labels. Most elevator datasets
annotate one `button` class with no text, which makes the assignment's required
OCR and end-to-end reports impossible to compute without hand-labelling
thousands of crops. This dataset's class names *are* the button labels, which
supplies that ground truth for free.

---

## Dataset used

**Source:** [Elevator Button Recognition, Sun Moon University](https://universe.roboflow.com/sun-moon-university/elevator-button-recognition-wrok1) (CC BY 4.0).

This is the standard benchmark in the elevator-button literature — the dataset
released with the OCR-RCNN papers (Zhu et al.), published as 2,100 panels /
21,767 labels / 365 classes. Our copy measures 2,019 / 22,843 / 368 with 154
single-instance classes, matching the published class-imbalance profile.
Attribution belongs to Sun Moon University; the `andrey-rel` copy is a
derivative re-upload.

### The two folders are the same photographs

The repository shipped `Dataset/` and `Dataset 2/`. An audit established they
are two exports of one collection:

| Evidence | Result |
|---|---|
| Filename-stem overlap | **2,019 / 2,019 (100%)** |
| Box IoU for the same object | mean **0.9999** |
| `Dataset/` downscaled to 416 vs `Dataset 2/` native 416 | mean pixel diff **1.24 / 255** |
| Content-only nearest-neighbour match (filenames ignored) | agrees on **2,017 / 2,019**, mean similarity 0.99991 |

Each export has what the other lacks:

| | `Dataset/` | `Dataset 2/` |
|---|---|---|
| Resolution | **640×640** (1.50× more real detail) | 416×416 |
| Label format | YOLOv8-seg polygons | **YOLOv8 bbox** |
| `data.yaml` | **Corrupt** — class names are README prose | **Clean** — 368 real labels |
| Annotations | 22,266 (578 missing) | **22,843** |

`Dataset/`'s `data.yaml` lists `'Roboflow is an end-to-end computer vision
platform that helps you'` as a class name; its third most common class (1,139
instances) maps to `'- use active learning to improve your dataset over time'`.
The class *IDs* are sound — only the dictionary is broken.

**The pipeline uses `Dataset/`'s sharper images with `Dataset 2/`'s correct
labels.** Because YOLO coordinates are normalised and the photographs are
identical, the label files apply with no transformation.

### Counts

| Quantity | Value |
|---|---|
| Images | 2,007 (12 dropped, no buttons left after filtering) |
| Button instances | 20,513 |
| OCR-scorable (readable text) | 13,938 |
| Symbol / unreadable buttons | 6,575 |
| Dropped as non-buttons | 2,330 |
| Distinct label strings | 258 |
| Split (seed 42) | 1,405 train / 401 valid / 201 test |

---

## Dataset preparation

`prepare_dataset.py` is deterministic and reproducible.

**1. Pair the two exports, and verify the pairing.** Matching is by filename
stem — Roboflow appends a fresh `.rf.<hash>` per export, so full filenames never
match — and every pair is then asserted by 64×64 grayscale cosine similarity
≥ 0.99. A bad pairing then fails loudly rather than quietly putting boxes on the
wrong photograph. All 2,019 pairs pass.

**2. Drop non-buttons (2,330 instances).** Decided by inspecting crops, not by
guessing from names:

| Class | n | Why |
|---|---|---|
| `led` | 475 | Seven-segment floor displays. **Actively dangerous if kept** — the recognizer would read a display as a button label and send the robot to a floor with no button. |
| `text`, `text_*` | 673 | Printed placards ("OPEN DOOR", "CAR CALL CANCEL"). Same hazard. |
| `keyhole`, `key`, `switch`, … | 931 | Key cylinders and toggles — not pressable. |
| `indicator`, `light`, `speaker`, `hat` | 251 | Lamps, grilles, emblems. |

**3. Keep symbol buttons for detection, exclude them from OCR scoring
(6,575).** `empty`, `blur`, `unknown` are genuinely pressable buttons that are
unmarked, out of focus or illegible; `alarm`, `open`, `close`, `up`, `down`,
`call`, `stop`, `fan` carry symbols. A button detector must find them, but
scoring them as OCR failures would understate the recognizer.

Three classes were reclassified after inspecting crops: `CH` is the wheelchair
pictogram ♿ (not the letters "CH"), `s` is a star glyph ★, and `-` is a bar
glyph. Their class names are mnemonics, not printed text. Scoring them would
have deflated reported OCR accuracy by ~1.5 points for free.

**4. Collapse to a single `button` class** for detection. Rationale in
[Model architecture](#model-architecture).

**5. Duplicate-aware splitting.** The shipped splits contained 4 duplicate image
pairs, **all four straddling a train/test or train/valid boundary** — real
leakage, invisible to a filename-based check because the duplicates have
different names. Images are clustered by content at cosine 0.95 and split at
*group* level, so duplicates can never separate. 12 groups (24 images) are held
together.

**6. Emit `ocr_ground_truth.json`** carrying the original label for all 20,513
boxes. This is what makes the OCR and end-to-end reports auto-generatable.

Verified after generation: every image has exactly one label file; all 20,513
instances are class `0`; zero malformed lines; zero out-of-bounds coordinates;
zero duplicate groups straddling a split.

---

## Model architecture

**Detector — YOLOv8n.** Anchor-free decoupled head, CSPDarknet backbone, PANet
neck. 3,005,843 parameters, 8.1 GFLOPs, 6.3 MB. Fine-tuned from COCO-pretrained
`yolov8n.pt` (standard transfer learning — not a pretrained elevator-button
detector). Chosen at "nano" scale because the target is a robot doing this on
embedded hardware in real time.

**Single-class detection**, deliberately:

1. The assignment's example output shows `"text": "27"` produced by the OCR
   stage, not by a detector class head. Detect-then-read is the architecture the
   brief describes.
2. Class imbalance is severe — 154 of 368 classes have exactly one instance, and
   the top 80 cover 95.6%. A 368-class detector would overfit head classes and
   fail on the tail.
3. Generalisation to unseen panels is a hard requirement. "Is this a button?" is
   a far more transferable function than "is this specifically a floor-47
   button?".

One consequence: with a single class, horizontal flip augmentation is safe,
since a mirrored button is still a button. It would corrupt digit identity in a
multi-class variant, and it stays disabled for the recognizer.

**Recognizer — hybrid.** YOLOv8n-cls (1,765,378 parameters, 3.3 GFLOPs) as
primary, PaddleOCR as fallback when classifier confidence < 0.55.

---

## Training configuration

Detector, from `reports/training_metadata.json`:

```
base model : yolov8n.pt (COCO-pretrained)
epochs     : 100 / 100  (no early stopping)
imgsz      : 640         (native dataset resolution — no resampling)
batch      : 16
optimizer  : auto (Ultralytics: AdamW)
seed       : 42
wall clock : 13m 34s
hardware   : NVIDIA GeForce RTX 5070 Ti, torch 2.11.0+cu128
```

Recognizer:

```
base model : yolov8n-cls.pt
epochs     : 120
imgsz      : 128         (crops exported at 128px — see below)
batch      : 96
classes    : 228 label strings present in training
train crops: 9,806  (12,553 including valid)
```

Crop resolution mattered more than any other change here. 64px crops at
`imgsz=64` for 60 epochs gave 48.2%; 128px crops at `imgsz=128` for 120 epochs
gave 66.9%.

---

## Hyperparameters

All live in [`config.yaml`](config.yaml) — they are not duplicated here, so they
cannot drift. Values that were tuned empirically rather than chosen:

| Parameter | Value | Why |
|---|---|---|
| `crop_padding_ratio` | **−0.22** | Negative = crop *inward*. Ground-truth boxes bound the whole button including its circular bezel, and recognizers read that bezel as `0`. Swept `{-0.30 … +0.15}`; −0.22 won for both EasyOCR and PaddleOCR, nearly tripling accuracy (9.1% → 26.4% on the sweep set). |
| `interpolation` | **linear** | `INTER_LINEAR` measured 5.6 points better than `INTER_CUBIC` for PaddleOCR (59.3% vs 53.7%). |
| `allowlist` | 34 chars | Derived from the 258 real ground-truth strings, not guessed. |
| `classifier_threshold` | 0.55 | Below this, consult PaddleOCR too. |
| `fliplr` (detector) | 0.5 | Safe only because detection is single-class. |
| `fliplr` (recognizer) | 0.0 | Mirroring changes digit identity. |

CLAHE contrast enhancement was tested and **removed** — it lowered PaddleOCR
accuracy to 55.3%.

---

## Evaluation results

Full numbers with failure-mode breakdowns:
**[`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md)**.

### Detection (held-out test split, 201 images / 2,046 instances)

| Metric | Value |
|---|---|
| Precision | **0.9111** |
| Recall | **0.9063** |
| mAP@50 | **0.9489** |
| mAP@50:95 | **0.6804** |
| Inference latency | **7.46 ms** (134 FPS), batch=1 |

Latency is measured over 120 images at batch=1 after 10 warm-up runs
that are discarded; mean / median / p95 = 7.46 / 6.66 /
9.05 ms on an NVIDIA GeForce RTX 5070 Ti. The full pipeline including
recognition is 1769 ms, so detection is roughly
237× cheaper than reading the buttons.

Test tracks validation closely (val mAP@50 0.9463, test 0.9489). Leakage would
show up as test scoring well above val, so this is a reasonable sign the split
is clean.

### Recognition — seven approaches on identical ground-truth crops

Every permitted category in the assignment's Part 3 was measured on the same
1,385 test crops:

| Approach | Accuracy | ms/crop | 1-char | 2-char |
|---|---|---|---|---|
| **YOLOv8n-cls (chosen)** | **66.9%** | **4.7** | **74.3%** | **55.2%** |
| PaddleOCR (PP-OCRv6 rec) | 59.4% | 153.6 | 65.9% | 48.7% |
| HOG + SVM (OpenCV) | 32.1% | 5.7 | 47.2% | 7.3% |
| EasyOCR | 26.4% | 19.7 | 34.4% | 13.5% |
| kNN on pixels (OpenCV) | 24.8% | 0.2 | — | — |
| Tesseract | 22.2% | 726.2 | 27.9% | 12.9% |
| Template matching (OpenCV) | 1.7% | 0.1 | — | — |

For reference, [OCR-RCNN](https://ieeexplore.ieee.org/document/9324975/) reports
**79.6%** character accuracy on this same dataset using a purpose-built
attention-RNN recognizer.

### Full pipeline (the assignment's three required sections)

Detection boxes, not ground-truth boxes — this is the real system end to end.

```
OCR
  Number of buttons evaluated: 1323
  Correctly recognized:        877
  Incorrectly recognized:      446
  OCR accuracy:                66.3%

End-to-End
  Correct floor identification: 861
  Total test cases:             1385
  Accuracy:                     62.2%
```

OCR on *detected* boxes (66.3%) essentially matches the ceiling measured on
*perfect* ground-truth crops (66.9%), which says detector box jitter costs
almost nothing — the boxes are tight enough that the recognizer barely notices.

| Button size | OCR accuracy |
|---|---|
| tiny (<24px) | 24.3% |
| small (24–32) | 60.0% |
| med (32–48) | 69.4% |
| large (48–64) | 75.8% |
| **xl (≥64)** | **79.4%** |

The largest bucket reaches **79.4%**, level with OCR-RCNN's published 79.6%
overall. The entire gap to that benchmark is the small-button tail.

### Failure modes

| Failure mode | Count |
|---|---|
| detected, OCR misread | 418 |
| button not detected | 59 |
| wrong button selected | 44 |
| detected, OCR failed (no read) | 3 |

Recognition, not detection, is the bottleneck: 418 of 524 failures are misreads
against only 59 missed buttons.

### Generalization to real photographs

The test split above shares a collection process and preprocessing pipeline with
training. To test the assignment's "panels not present in the training dataset"
requirement properly, the pipeline was run on **8 elevator panels photographed
on a phone** — full resolution, portrait, WhatsApp-compressed, real glare.
Annotated outputs and per-image notes are in
[`reports/unseen/`](reports/unseen/); there are no ground-truth annotations, so
every count was checked by eye against the annotated image.

Detection transferred well. 7 of 8 panels were detected essentially completely,
**159 buttons** in total, at resolutions from 720×1280 to 1200×1600 and in
portrait aspect, from a model trained only on 640×640 squares.

Recognition split into two regimes, and the overall average describes neither
of them:

| Panel type | Panels | Text buttons | Correct | Accuracy |
|---|---|---|---|---|
| Label printed **on** the button | 5 | 73 | 65 | **89.0%** |
| Label printed **beside** the button | 2 | 54 | 0 | **0%** |
| Flat capacitive touch panel | 1 | 4 | 0 | 0% (nothing detected) |
| **All** | **8** | **131** | **65** | **49.6%** |

Two panels were read perfectly: **29/29** on a 27-floor panel and **10/10** on
another. Where the design matches training, real-world accuracy (89.0%) is
*higher* than the in-dataset figure (66.3%) — those photos are sharper than the
416-to-640 upscaled benchmark images.

The three failures are systematic rather than random:

**1. Labels beside the button, not on it.** Two panels engrave the floor number
to the *left* of each button. `crop_padding_ratio = -0.22` crops inward — the
setting that nearly tripled accuracy on the benchmark, because there every label
sits on its button — and here it removes the number entirely, leaving a blank
metal disc. 0 of 54 correct. The tuning is overfitted to one panel convention.

**2. Symbol buttons always get a spurious text label.** A fan reads as `5`,
STOP as `B3`, a bell as `2`, door-open arrows as `11`. The classifier was trained
only on OCR-scorable crops, so it has no symbol class and no way to abstain:
forced to choose among 228 text labels, it picks one confidently. This is a
design gap rather than a tuning problem, and it has real consequences: a robot
asked for floor 5 could press the fan.

**3. Flat capacitive panels are invisible to the detector.** One panel is a
printed touch surface with no raised buttons. Zero detections — the detector
learned bezels and edges, and there are none.

None of these three appear anywhere in the 62.2% end-to-end figure, because
every benchmark image is a preprocessed square of a physical-button panel.

---

## OCR approach

Part 3 asks for a justification. The honest one is that this is not really an
OCR problem, and the measurements above are what showed it.

### Why general OCR engines struggle here

1. **A third of buttons are physically unreadable.** Median box is 42×42 px;
   31.4% are under 32×32; the smallest 5% are ~18 px. Upscaling interpolates —
   it does not recover detail the camera never captured. Every approach tested
   scores near zero on the `tiny(<24px)` bucket.
2. **OCR engines read *sequences*; a button has none.** EasyOCR and PaddleOCR
   are CRNNs: a CNN extracts features, then an **RNN reads them left-to-right**.
   That recurrent stage exists to exploit context — seeing `R-E-C-E-I-P` makes
   `T` likely. Given a lone `5` there is no context at all, so the single most
   valuable component is inapplicable, and a model expecting sequences can
   hallucinate structure that is not there.
3. **The domain is close to worst case.** Engraved metal means the text is *the
   same colour as its background*, separated only by a shadow in the groove.
   Add backlit plastic, specular glare from ceiling lights, and surfaces
   thousands of people touch daily.
4. **The button's own shape fights the recognizer.** Buttons are round, and a
   circle drawn around a character reads as `0`. At neutral padding, 11 of the
   first 220 buttons read as `"0"` regardless of what was printed.
5. **Two-character labels get half the pixels.** A two-digit button is not a
   bigger button — median box width is 35.4 px for two characters vs 41.5 px for
   one, so each glyph gets **17.7 px instead of 41.5 px**. This is why floors
   10+ are markedly harder than floors 1–9.

### Why a hybrid

Reframing the task as **closed-set classification** — pick one of the label
strings seen in training, rather than decode an arbitrary character sequence —
is both easier and better suited to the data, and it is what the OCR-RCNN
literature does for this exact problem. It is the assignment's fifth permitted
option, "a lightweight vision/OCR model".

But a classifier has no output slot for a label it never saw. Measured: 11 of
1,385 test buttons (0.8%) carry a label absent from training, so the closed-set
ceiling here is 99.2% — small, but on a genuinely novel panel it matters, and
the brief requires working on unseen panels.

So: **classifier first, PaddleOCR when the classifier is unsure.** The
classifier is more accurate and 33× faster; PaddleOCR reads arbitrary strings
and covers the closed-set gap. Each half covers the other's weakness.

Two implementation details mattered more than expected:

- **Use `recognize()`, not `readtext()`.** `readtext()` runs CRAFT text
  *detection* first, which found nothing in **71%** of small crops. We already
  know where the text is — it is the whole crop.
- **PaddleOCR's detection model is bypassed entirely.** Besides being
  unnecessary, it crashes with a oneDNN `NotImplementedError` on this machine.
  Only `TextRecognition` is used.

---

## Floor-selection logic

`src/floor_selector.py` covers the assignment's "build useful post-processing
logic".

**Grid inference.** Elevator buttons sit on a near-regular grid and floor
numbers ascend predictably. Detections are clustered into columns using a
tolerance derived from median button width; within a column, if the buttons
above and below a gap carry consecutive numbers and the vertical spacing is
consistent, the unread button between them is inferred. Given that recognition
tops out near 67% and unread buttons are usually unread because of *glare*
rather than absence, this recovers buttons that would
otherwise be lost. Anything inferred is marked
`inferred: true` with reduced confidence (0.35) and is **never** presented as a
direct reading.

**Matching cascade:**

1. Normalise the target through the *same* function used on OCR output —
   comparing a raw target against normalised readings is a classic silent
   failure.
2. **Exact match** → return it.
3. **Multiple exact matches** → highest combined confidence, flagged
   `ambiguous`.
4. **Fuzzy match** at Levenshtein ≤ 1, with two restrictions, returned with
   `match_type: "fuzzy"` so the caller knows it was inexact. Substitutions only,
   length preserved: `Levenshtein("20","2")` is 1, so plain edit distance
   answered a query for floor 20 with the button labelled `2`. And a candidate
   that is itself a known label is never offered for a different target, since a
   button reading `22` means floor 22 rather than a misread `20`. Fuzzy matching
   is there to rescue garbage readings, not to swap one real floor for another.
5. **Nothing plausible** → a null result with a machine-readable reason
   (`not_detected` / `ocr_failed` / `no_match`). **It never fabricates a box.**

**Confidence semantics:** `confidence = detection_conf × ocr_conf`, so the
number reflects end-to-end certainty that this is the right button, not merely
that something was detected.

**No blanket digit coercion.** An early version mapped lookalike letters onto
digits inside otherwise-numeric strings. It did real damage: `normalize("B1")`
returned `81`, `G2` became `62`, `L1` became `11`. All of those are genuine
labels here (`B1` alone appears 150 times), and because the query is normalised
through the same function, every basement and lobby lookup silently missed.

The only characters mapped now are `Q`, `|` and `!`, none of which is ever a
label on its own. Glyph confusion is handled instead by `snap_to_vocabulary()`,
which compares a reading against the 258 known label strings and so can only
ever move it *onto* a real label, never away from one.

---

## Inference instructions

```bash
# One floor — prints ONLY the required JSON on stdout
python detect_floor.py --image panel.jpg --floor 27
python detect_floor.py --image panel.jpg --floor B1 --visualize
python detect_floor.py --image panel.jpg --floor 27 | jq .button.center

# Every button on a panel
python inference.py --image panel.jpg
python inference.py --dir examples/input --visualize --json-out examples/output

# Reproduce the reports
python evaluate.py                       # the three required sections
python ocr_compare.py                    # EasyOCR / Tesseract / PaddleOCR
python opencv_baseline.py                # template match / kNN / HOG+SVM
python eval_classifier.py                # the trained classifier
python tests/test_coordinate_space.py    # bbox coordinate-space check
```

`detect_floor.py` exit codes: `0` found, `3` no matching button, `1` bad input.

`--floor` accepts numeric and alphanumeric labels: `27`, `B1`, `G`, `LG2`, `P3`.

### Coordinate space

`bbox` is `[x1, y1, x2, y2]` in **absolute pixels of the original input image**,
at any resolution or aspect ratio. This is verified by
`tests/test_coordinate_space.py`, which pastes a known panel into canvases from
640×640 to 3024×4032 and checks the boxes land on the right buttons.

This test matters because **the entire test split would pass regardless** —
every dataset image is a perfect 640×640 square, and square input receives no
letterbox padding, so a letterbox coordinate bug would be completely invisible
on it while breaking every real photograph.

---

## Known limitations

- **Recognition is the bottleneck, not detection.** Detection reaches 0.95
  mAP@50; recognition tops out at 66.9% on perfect crops. The pipeline finds
  buttons reliably and reads them unreliably.
- **Button size dominates everything.** 31.4% of buttons are under 32×32 px and
  accuracy on the smallest bucket is near zero for *every* method tested.
- **Two- and three-character labels are much harder** (55.2% vs 74.3%) because
  each glyph gets roughly half the pixels. Floors 10+ are the common case in
  tall buildings.
- **The classifier is closed-set** — it cannot read a label absent from
  training. 0.8% of test buttons, but higher on a genuinely novel panel. The
  PaddleOCR fallback mitigates rather than solves this.
- **Single-source training data.** All 2,019 images come from one collection
  with one preprocessing pipeline (auto-orient, stretch to square, JPEG). Panel
  design diversity is limited.
- **Detection range.** At extreme aspect ratios (1:3) the panel letterboxes down
  far enough that recall falls to ~56%. Coordinates stay correct; the buttons
  simply become too small for the detector.
- **Grid inference only fills gaps; it does not correct misreads.** If a button
  is unread, its neighbours can supply the value. But if it is read *wrongly* —
  a `24` that comes back `21` in a column running 36, 33, 30, 27, 24 — the
  spatial pattern contradicts the reading and nothing acts on it. Since misreads
  are 418 of 524 end-to-end failures and confident misreads are the common case,
  this is the single most valuable unbuilt piece of post-processing.
- **Grid inference assumes a regular layout.** It will not help on circular,
  staggered or touchscreen panels, and it only infers *numeric* runs.
- **No pressed/illuminated state handling.** The system cannot tell whether a
  button is already lit.
- **`Final dataset/data.yaml` contains an absolute path** (Ultralytics requires
  it). It is regenerated on every `prepare_dataset.py` run.
- **The recognizer cannot abstain on symbol buttons.** It was trained only on
  crops carrying readable text, so it has no symbol class. Shown a fan or a
  STOP button it still returns a floor label, confidently. On real photographs
  this was the most common error, and it is the one with real-world
  consequences — a robot asked for floor 5 could press the fan.
- **Crop geometry is tuned to one panel convention.** `crop_padding_ratio`
  of −0.22 assumes the label is printed *on* the button. On panels that engrave
  the number *beside* the button it removes the label entirely: 0 of 54 correct
  across two such panels.
- **Flat capacitive panels are not detected at all.** The detector learned
  raised buttons with bezels; a printed touch surface offers nothing to find.

---

## What I would improve with more time

1. **Fine-tune PaddleOCR's recognition model on our 12,553 crops.** The highest
   expected gain and the clearest route toward OCR-RCNN's 79.6%. It keeps
   character-level reading — so it handles unseen labels, unlike the classifier
   — while gaining in-domain adaptation to glare and engraving.
   [PaddleOCR's docs](https://www.paddleocr.ai/v2.9/en/ppocr/model_train/finetune.html)
   state ≥5,000 real samples "greatly improves" results; we have 2.5× that, plus
   a 34-character dictionary instead of the stock ~95.
2. **Scene-text super-resolution before recognition.**
   [TSRN](https://arxiv.org/abs/2005.03341) improves CRNN accuracy ~13% on
   low-resolution text — directly targeting our binding constraint.
3. **Add a `symbol` class to the recognizer.** The 6,575 symbol crops already
   exist and are already labelled — they were excluded from the classifier's
   training set, which is exactly why it cannot abstain. Adding one extra class
   is cheap and removes the most dangerous real-world error.
4. **Detect the label region rather than assuming it is on the button.** Fixed
   inward cropping fails whenever the number sits beside the button. Locating
   the text relative to each detection, instead of assuming it is centred,
   would fix a whole panel family at once.
5. **Broaden the training data to other panel conventions** — beside-button
   labels and flat capacitive surfaces are both absent from the benchmark and
   both common in the field.
6. **Perspective correction via button-corner detection**
   ([arXiv:2007.11806](https://arxiv.org/pdf/2007.11806)) before cropping, so
   angled panels present flat glyphs to the recognizer.
7. **Multi-frame temporal voting.** A robot sees a video, not one frame. Voting
   across frames would recover most glare-induced failures for free.
8. **Make grid inference correct misreads, not just fill gaps.** A reading that
   contradicts its column's arithmetic is almost certainly wrong, and misreads
   are the dominant failure mode. Extending it to alphanumeric runs
   (`B3, B2, B1, G, 1, 2`) would help too.
9. **Quantisation / TensorRT export** for embedded deployment.

---

## Repository layout

```
├── README.md                this file
├── requirements.txt         dependencies, plus the cu128 GPU note
├── config.yaml              every tunable value
├── .gitignore
│
├── detect_floor.py          PRIMARY DELIVERABLE: target floor -> JSON
├── inference.py             all buttons on a panel
├── train.py                 trains the YOLOv8n detector
├── evaluate.py              writes reports/EVALUATION_REPORT.md
│
├── prepare_dataset.py       merges both raw exports -> Final dataset/
├── export_crops.py          button crops -> classification dataset
├── train_classifier.py      trains the YOLOv8n-cls recognizer
│
├── ocr_compare.py           EasyOCR vs Tesseract vs PaddleOCR
├── opencv_baseline.py       template matching, kNN, HOG+SVM
├── eval_classifier.py       scores the classifier on the test split
├── ocr_gate.py              recognition ceiling on ground-truth crops
│
├── models/
│   ├── best.pt              trained detector
│   └── button_cls.pt        trained recognizer
├── src/
│   ├── detector.py          YOLO wrapper, Detection dataclass
│   ├── ocr.py               all five recognizers + normalisation
│   ├── floor_selector.py    grid inference + matching cascade
│   ├── pipeline.py          ties the three stages together
│   └── viz.py               annotated output images
├── tests/
│   └── test_coordinate_space.py
├── reports/                 metrics, EVALUATION_REPORT.md, unseen/
├── examples/input|output/   sample panels and their results
└── Unseen Dataset/          8 phone photos used for the external test
```

The four analysis scripts in the third group are not part of the pipeline. They
produce the numbers behind the engine comparison, and are kept so those numbers
can be reproduced rather than taken on trust.

## License and attribution

Dataset: **Elevator Button Recognition**, Sun Moon University, CC BY 4.0 —
the benchmark released with the OCR-RCNN work of Zhu et al.
