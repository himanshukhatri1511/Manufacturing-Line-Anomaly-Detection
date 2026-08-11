# Manufacturing Line Anomaly Detection — Computer Vision + Statistical Process Control

Detecting anomalies in a repetitive manual assembly task using two independent computer-vision
approaches built on the same video dataset: an **event-based pipeline** (Phases 1–2) that checks
whether the right steps happened in the right order and within the right time, and a **shape-based
pipeline** (Phase 5) that checks whether the entire motion pattern of a cycle looks statistically
normal, moment by moment. The second approach is a standalone proof-of-concept, not a patch on the
first — see [How the two approaches differ](#how-the-two-approaches-differ) below.

## What this project does

The dataset is a fixed-camera recording of a worker repeating the same manual assembly task, split
into **78 cycles**, each cycle made up of **7 short task videos** recorded in sequence
(`Center_Assembly_Jig → Right_Screw_Feeder → Center_Assembly_Jig`, roughly). The notebook:

1. Tracks motion inside three fixed workstation zones (material bin, screw feeder, assembly jig)
   using classical background subtraction — no manual labeling required.
2. Builds a rule-based system (Markov transition matrix + Gaussian duration model, YOLOv8-verified)
   that flags cycles with the wrong step order, missing steps, or abnormal timing.
3. Builds a second, independent statistical system that layers all 78 cycles' motion curves on top
   of each other, learns the "normal" shape and variation per workstation zone, and flags cycles
   whose motion pattern deviates from it — catching things like hesitation or extra fumbling that
   the first system cannot see, because it never checks the *shape* of the motion.
4. Validates the second system with a proper train/test split and a synthetic anomaly injection
   suite (missing task, scrambled task order, rushed/partial cycle).

## Repository contents

| File | Description |
|---|---|
| `Manufacturing_Anomaly_Detection_Phase5_Fixed.ipynb` | The full notebook: Phases 1–2 (event-based pipeline) followed by Phase 5 and Phase 5 Part 2 (shape-based pipeline) |

## Dataset structure this notebook expects

```
VideoDataset/
└── Cycles/
    ├── Cycle_0/
    │   ├── task_1.mp4
    │   ├── task_2.mp4
    │   ├── ...
    │   └── task_7.mp4
    ├── Cycle_1/
    │   └── ...
    └── Cycle_77/
        └── ...
```

- Each `Cycle_N` folder holds the task videos for one full repetition of the process, in order.
- Videos are read with OpenCV (`cv2.VideoCapture`), so any codec OpenCV supports works.
- This dataset is not included in this repository (video files). Point the path variables below
  at your own copy, or adapt the loader cells if your folder layout differs.

## How to use this notebook

### 1. Install dependencies

```bash
pip install opencv-python numpy matplotlib pandas natsort scipy ultralytics
```

`ultralytics` downloads the YOLOv8n weights (`yolov8n.pt`, ~6 MB) automatically on first run —
an internet connection is needed the first time you run the YOLO cells.

### 2. Point the notebook at your dataset

Update the dataset path in **Cell 0** and **Cell 13** (search for `DATASET_PATH` and
`BASE_DATASET_PATH`) to point at your local `Cycles` folder, e.g.:

```python
DATASET_PATH = r"D:\your\path\to\VideoDataset\Cycles\Cycle_0"     # Cell 0, single-cycle exploration
BASE_DATASET_PATH = r"D:\your\path\to\VideoDataset\Cycles"        # Cell 13, full dataset
```

The workstation zone coordinates (`WORKSTATION_ROIS`, in Cells 2 and 13) are calibrated to this
specific camera angle — if you use your own footage, you'll need to re-draw these rectangles to
match your camera's frame.

### 3. Run top to bottom

The notebook is meant to be run in order, in one sitting, since later cells depend on variables
defined earlier (`WORKSTATION_ROIS`, `ManufacturingMotionEngine`, `normalized_curves`, etc.).
Runtime on the full 78-cycle dataset is dominated by video I/O and YOLO inference; expect it to
take a while the first time. `MAX_CYCLES_TO_PROCESS` (Cell 13) and `NUM_CYCLES_TO_LOAD` (Phase 5
Cell 1) can be lowered for a quick smoke test before running the full dataset.

### 4. What each section produces

| Section | Cells | Output |
|---|---|---|
| Setup + motion tracking | 0–7 | Per-zone motion-density curves for one cycle, smoothed and thresholded into on/off events |
| Rule-based cycle report | 8–9 | A single cycle's pass/fail report: sequence order, screw-grab count, cycle duration |
| YOLO verification | 10–12 | Confirms motion events correspond to a real detected person, not lighting/shadow noise |
| Mass feature extraction | 13 | Runs the pipeline over all 78 cycles, writes `manufacturing_ml_dataset.csv` |
| `IndustrialSequenceAI` | 14–16 | Learns a Markov transition matrix + Gaussian duration model from the CSV; scores new cycles |
| **Phase 5** | 18–33 | Layers all 78 cycles' motion curves; builds a mean ± 2σ "golden band" per zone; heatmap and dendrogram views |
| **Phase 5, Part 2** | 34–47 | Train/test split, duration check, `score_cycle_phase5()` classifier, held-out validation, synthetic anomaly test suite |

## How the two approaches differ

**Phase 1–2 (event-based):** the continuous motion signal is smoothed and thresholded into a
binary "zone active / not active" flag. Everything downstream — the Markov transition matrix, the
Gaussian duration model, YOLO verification — operates on these discrete events. This makes it fast
and robust to frame-level noise, but structurally blind to *how* an activation happened: a worker
who hesitates or fumbles but still finishes in the right order and within the normal time window
looks completely normal to this system.

**Phase 5 (shape-based):** the full continuous motion curve is kept, never collapsed into events.
Every cycle is resampled onto a common 0–100% "cycle progress" axis (since cycles vary in length)
and layered on top of the others per zone, producing a mean curve and a ±2σ normal-variation band —
the same idea as a Statistical Process Control (SPC) chart on a production line. A new cycle is
flagged if too much of its curve falls outside that band, or if its total duration is a statistical
outlier. This catches anomalies in motion *quality*, not just sequence and timing, at the cost of
needing careful statistical calibration (smoothing, a proper train/test split, and a tuned decision
threshold) since it's comparing raw signal rather than a pre-digested checklist.

They are complementary, not redundant: Phase 1–2 catches wrong order / missing steps / abnormal
timing; Phase 5 catches abnormal motion *within* an otherwise correctly-ordered, correctly-timed
step.

## Phase 5 validation results

Phase 5, Part 2 validates the shape-based classifier two ways:

**Held-out validation** — the "golden band" is built from ~85% of the 78 cycles only; the
remaining cycles are scored without ever being seen during baseline construction.

```
PASS: 10/11   FAIL: 1/11
False positive rate on unseen normal cycles: 9.1%
```

**Synthetic anomaly injection suite** — three known-bad variants are built by manipulating a real
cycle's task videos, then scored the same way:

| Test case | Result | Detail |
|---|---|---|
| Genuine unseen normal cycle | PASS | duration z-score = 0.12 |
| Missing task (one task video removed) | FAIL (correctly) | 21.4% of cycle out-of-band |
| Scrambled order (task videos reversed) | FAIL (correctly) | 13.4% of cycle out-of-band |
| Rushed/partial cycle (only first 2 of 7 tasks) | FAIL (correctly) | 13.6% out-of-band + duration z-score = 7.40 |

All three injected anomaly types were correctly flagged, each with a comfortable margin above the
8.0% decision threshold, while the held-out false-positive rate stayed in the single digits.

## Known limitations and future work

- **Linear time-normalization is sensitive to timing jitter.** Because every cycle is stretched
  onto a fixed 0–100% axis, a worker who reaches a zone slightly earlier or later than usual — a
  normal amount of human variation — can occasionally register as a shape deviation near the edge
  of a usage window. This was the source of most of the false positives encountered during
  calibration (see the notebook's markdown cells for the full debugging narrative).
- **Dynamic Time Warping (DTW)** is the natural next step: instead of a fixed linear stretch, DTW
  would let each cycle align to its own event timing before comparison, which should reduce
  timing-jitter false positives further — at the cost of significantly more implementation and
  computational complexity (DTW Barycenter Averaging would be needed to build the reference curve).
- The workstation ROI coordinates and `EXPECTED_SEQUENCE` are hard-coded for this specific camera
  setup and task; adapting this to a different line requires re-calibrating both.

## Requirements

- Python 3.9+
- `opencv-python`, `numpy`, `matplotlib`, `pandas`, `natsort`, `scipy`, `ultralytics`
- A local copy of the video dataset in the folder structure described above
