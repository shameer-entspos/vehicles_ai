# CCTV Vehicle Number Plate Scanner

A Python system that watches a CCTV stream (or a video file), finds every vehicle, reads its
**number plate**, and saves **one record per vehicle** into a database, together with the plate
text, a confidence score, the vehicle type, timestamps and snapshot images.

It runs the same way on a recorded video (for testing) and on a live RTSP camera. You change one
setting to switch between them.

---

## 1. What it does

- Reads video from a file, a webcam, or an RTSP/HTTP CCTV stream. A dropped stream reconnects automatically.
- Detects and tracks cars, motorcycles, buses and trucks, giving each vehicle a stable ID.
- Finds the number plate on each vehicle and reads it with OCR over many frames.
- Combines the reads by voting, so one blurry frame does not decide the answer.
- Saves one record per vehicle (no duplicates) in SQLite, plus plate and vehicle images.
- Optionally marks each vehicle as `entry` or `exit` when it crosses a line you draw across the gate.
- Shows a live preview with boxes, plate text and the latest records, and can save the annotated video.

---

## 2. How it is built

### 2.1 Pipeline

```
 CCTV / video file
        |
        v
 [1] VideoSource            src/source.py      read frames, auto-reconnect RTSP
        |
        v
 [2] VehicleTracker         src/detector.py    YOLOv8 detects vehicles, ByteTrack gives IDs
        |
        +--> per vehicle (track ID)
        |       |
        |       v
        |   [3] PlateReader        src/plate.py     plate detector -> crop -> EasyOCR
        |       |                  src/platefmt.py  clean text, fix O/0 I/1 S/5 B/8 by position
        |       v
        |   [4] TrackState         src/recorder.py  collect reads, vote, find consensus plate
        |       |
        |       v
        |   [5] Recorder           src/recorder.py  save / merge record, save snapshots
        |       |
        |       v
        |   [6] PlateDB            src/db.py        SQLite: data/plates.db
        |
        +--> [7] LineCounter (optional)  src/counter.py   entry / exit direction
```

`main.py` ties these together in one loop and draws the preview window.

### 2.2 Models and libraries used

| Part | What is used | Role | Where it comes from |
|---|---|---|---|
| Vehicle detection | **YOLOv8n** (`yolov8n.pt`), trained on COCO | Finds car, motorcycle, bus, truck (COCO classes 2, 3, 5, 7) | Downloaded automatically by Ultralytics on first run |
| Vehicle tracking | **ByteTrack** (`bytetrack.yaml`, built into Ultralytics) | Keeps the same ID for a vehicle across frames | Included with `ultralytics` |
| Plate detection | **A YOLO model trained on license plates** (`models/plate.pt`) | Locates the plate inside each vehicle crop | You provide it (see section 6) |
| Plate reading (OCR) | **EasyOCR** (CRAFT text detector + CRNN text recognizer) | Reads the characters on the plate crop | Downloaded automatically on first run |
| Image preprocessing | **OpenCV** | Grayscale, upscaling, CLAHE contrast enhancement, video I/O, drawing | `opencv-python` |
| Storage | **SQLite** (Python `sqlite3`) | Stores one record per vehicle | Built into Python |
| Configuration | **PyYAML** | Reads `config.yaml` | `pyyaml` |

Why this design:
- Vehicle detection uses a pretrained COCO model, so it needs no training.
- Plate detection is the part that needs a specialized model, because plate styles differ a lot between countries.
- If `models/plate.pt` is missing, the app still runs, but it OCRs the lower part of each vehicle instead. This is much less accurate.

### 2.3 How a number plate becomes a record

1. **Detect and track.** Each frame is passed to YOLOv8 with ByteTrack. Every vehicle gets an ID.
2. **Check size.** Vehicles shorter than `min_vehicle_height_px` are too far away to read and are skipped.
3. **Find the plate.** The vehicle crop goes to the plate detector. The best plate box is cropped with a small margin.
   If a plate crop is narrower than `min_plate_width_px`, it is skipped.
4. **Preprocess.** The plate is converted to grayscale, enlarged to about 160 px wide or more, and contrast-enhanced with CLAHE.
5. **OCR.** EasyOCR reads only `A-Z` and `0-9`. Two-row plates are read top row first, single-row plates left to right.
   Confidence is averaged and weighted by text length.
6. **Clean up.** The text is made uppercase with spaces and hyphens removed. Common OCR mix-ups are fixed by position:
   the first characters should be letters and the rest digits, so a `0` read in the letter part becomes `O`, and an `O` read in the digit part becomes `0`.
   The result is checked against `plate_patterns`. A read that matches gets full weight, and one that does not gets 0.4 weight
   (or is dropped if `strict_format: true`).
7. **Vote.** Reads for the same vehicle are grouped when they differ by at most one character
   (`LEA1234` and `LEA1284` count as the same plate). The group with the highest total weight wins.
8. **Save the record.**
   - **Early save:** when the winning plate has been read `confirm_reads` times with average confidence at least `confirm_conf`, the record is saved immediately and OCR stops for that vehicle.
   - **Save on exit:** otherwise it is saved when the vehicle has not been seen for `lost_frames` frames, or when you stop the program.
9. **Status.** `confirmed` means at least `min_reads_accept` agreeing reads, or one read above `high_conf_accept`.
   `low_conf` means a plate was guessed from weak reads. `unreadable` means the vehicle was seen clearly but no plate could be read
   (turn off with `log_unreadable: false`).
10. **No duplicates.** If the same plate (or a one-character variant) appears within `cooldown_sec`, it is merged into the existing record,
    even if the tracker gave the vehicle a new ID. A `low_conf` record is upgraded if a later read is `confirmed`.

### 2.4 Optional entry / exit direction

If `line` is set in `config.yaml`, a red line is drawn across the frame. A vehicle's bottom-center point is tracked frame to frame.
When it crosses the line, the crossing direction is stored as `entry` or `exit` in the `direction` column. Set `line: null` to disable this.
It only adds information. The record itself is created from the plate read, not from the line.

---

## 3. Project structure

```
anpr_project/
├── main.py                 App entry point: main loop, preview window, args
├── config.yaml             All settings
├── pyproject.toml          Project and dependency list (used by uv)
├── uv.lock                 Locked package versions (created by uv)
├── .python-version         Python version used by uv
├── requirements.txt        Same dependencies as a plain list (used by pip)
├── .gitignore
├── yolov8n.pt              Vehicle detector weights (downloaded automatically on first run)
├── README.md               This file
├── src/
│   ├── source.py           Video file / webcam / RTSP input with auto-reconnect
│   ├── detector.py         YOLOv8 vehicle detection + ByteTrack tracking
│   ├── plate.py            Plate detector + EasyOCR reader
│   ├── platefmt.py         Plate text cleanup, position-based fixes, similarity check
│   ├── recorder.py         Per-vehicle voting, record creation, de-duplication
│   ├── counter.py          Entry / exit line-crossing logic
│   └── db.py               SQLite storage
├── scripts/
│   ├── export_csv.py       Print / search records and export to CSV
│   └── download_samples.py Download sample test videos
├── tests/
│   └── test_logic.py       Tests that need no GPU or models
├── models/                 Put your plate detector here as plate.pt (create this folder)
├── videos/                 Test videos
├── .venv/                  Virtual environment (created by uv or pip, not committed)
└── data/                   Created at run time
    ├── plates.db           The records (SQLite)
    └── snapshots/          Vehicle and plate images
```

---

## 4. Setup

Use **either** Option A (uv) **or** Option B (pip). You do not need both.

### Before you start

1. **Python version.** Use Python 3.10, 3.11 or 3.12 (3.11 recommended). Very new versions (3.13 and later) may not have
   packages for PyTorch and EasyOCR yet.
2. **Create the models folder** for the plate detector:
   ```bash
   mkdir models
   ```
   Then copy your plate detector into it as `models/plate.pt` (see section 6).
3. **Check the video path.** `config.yaml` defaults to `videos/car-detection.mp4`. Either put that file in `videos/`,
   edit `source:` in `config.yaml`, or pass `--source videos/your_video.mp4` when you run.

### Option A: uv only

[uv](https://docs.astral.sh/uv/) manages the Python version, the virtual environment and the packages.

```bash
# 1. Install uv once
#    Linux / macOS:  curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows:        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Pin Python 3.11 for this project (updates .python-version)
uv python pin 3.11

# 3. Create .venv and install everything from pyproject.toml / uv.lock
uv sync

# 4. Test and run (no need to activate .venv when you use "uv run")
uv run python tests/test_logic.py
uv run python main.py --source videos/pk_plates_1.mp4
```

`pyproject.toml` must list the dependencies. If your `dependencies = []` is empty, use this:

```toml
[project]
name = "anpr-project"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "ultralytics>=8.3.0",
    "opencv-python>=4.9",
    "easyocr>=1.7.1",
    "pyyaml>=6.0",
    "numpy",
]
```

Or let uv fill it in for you: `uv add -r requirements.txt`.

**NVIDIA GPU with uv.** Install the CUDA build of PyTorch, then run with `--no-sync` so uv does not replace it
with the CPU build. Also set `device: "0"` and `ocr_gpu: true` in `config.yaml`.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run --no-sync python main.py
```

### Option B: pip only

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # Linux / macOS

# 2. Install the packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Test and run
python tests/test_logic.py
python main.py --source videos/pk_plates_1.mp4
```

**NVIDIA GPU with pip.** Run this before the requirements install, then set `device: "0"` and `ocr_gpu: true` in `config.yaml`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Check the install

```bash
uv run python -c "import ultralytics, easyocr, cv2; print('all good')"     # uv
python -c "import ultralytics, easyocr, cv2; print('all good')"            # pip
```

### Notes

- **First run needs internet once.** `yolov8n.pt` and the EasyOCR model files are downloaded automatically.
  After that it works offline. `yolov8n.pt` is saved in the folder you run the command from, so run from the project root.
- **Linux server with no display.** Use `--no-show`. If OpenCV complains about missing libraries, install the headless build:
  `pip uninstall opencv-python && pip install opencv-python-headless` (with uv: `uv pip uninstall opencv-python && uv pip install opencv-python-headless`).
- **Suggested `.gitignore`:**
  ```
  .venv/
  data/
  __pycache__/
  *.pt
  videos/*.mp4
  ```

---

## 5. Run

The commands below use `python`. With uv and no activated environment, put `uv run` in front, for example `uv run python main.py`.

```bash
python tests/test_logic.py                         # quick check, no models needed
python main.py                                     # uses the source in config.yaml
python main.py --source videos/pk_plates_2.mp4     # a video file
python main.py --source videos/test.mp4 --save-video out.mp4
python main.py --no-ocr                            # vehicles only, much faster
```

Live CCTV:

```bash
python main.py --source "rtsp://admin:PASSWORD@192.168.1.64:554/Streaming/Channels/101" --no-show
```

Check the RTSP URL opens in VLC first. Typical formats:
- Hikvision: `rtsp://admin:PASS@IP:554/Streaming/Channels/101`
- Dahua: `rtsp://admin:PASS@IP:554/cam/realmonitor?channel=1&subtype=0`

Command-line options: `--config`, `--source`, `--no-show`, `--no-ocr`, `--save-video`.

**Preview window:** green box = record saved, light-blue box = still reading, yellow box = detected plate, red line = entry line,
bottom-left = latest records. Press **q** to quit. When you quit, vehicles still in view are saved too.

### Where the results are saved

- Records: `data/plates.db`, table `plate_records`
- Images: `data/snapshots/` (`..._vehicle.jpg` and `..._plate.jpg`)

```bash
python scripts/export_csv.py                  # table in the terminal + data/plates.csv
python scripts/export_csv.py --plate LEA      # search by part of a plate
sqlite3 data/plates.db "SELECT plate, status, first_seen FROM plate_records"
```

Plates are stored as capital letters and digits only, so `LEA-1234` is saved as `LEA1234`.

### Database table `plate_records`

| Column | Meaning |
|---|---|
| `id` | Record number |
| `plate` | Plate text, empty if unreadable |
| `plate_conf` | Average OCR confidence (0 to 1) |
| `status` | `confirmed`, `low_conf` or `unreadable` |
| `vehicle_type` | car, motorcycle, bus, truck |
| `camera_id` | From `config.yaml` |
| `direction` | `entry`, `exit` or empty |
| `first_seen`, `last_seen` | Clock time when the vehicle was first and last recorded |
| `reads` | Number of agreeing OCR reads |
| `track_id` | Tracker ID of the vehicle |
| `video_sec` | Position in the video (seconds), useful for file tests |
| `vehicle_image`, `plate_image` | Paths to the saved images |

---

## 6. Plate detector model (important)

`config.yaml` expects the plate detector at `models/plate.pt`. Without it, accuracy drops a lot.

**Quick start.** Download a pretrained YOLOv8 plate detector and save it as `models/plate.pt`. Two options on Hugging Face:
- `Murd0ck/LicensePlateDetector_YOLOv8n` (trained on Ukrainian plates)
- `Babblu2821/alpr-plate-detector` (Indian and European plates)

Neither was trained on Pakistani plates, so expect mixed results on them.

**Best accuracy for your own plates.** Fine-tune on a Pakistani plate dataset (Roboflow Universe, search "Pakistani License Plates",
CC BY 4.0) plus frames from your own camera. Aim for 500 to 2000 labeled images from your camera angle.

```bash
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640
# then copy runs/detect/train/weights/best.pt to models/plate.pt
```

Frames from your actual camera help more than any generic dataset, because angle, lighting and plate style matter a lot.

---

## 7. Configuration reference (`config.yaml`)

| Setting | Default | What it does |
|---|---|---|
| `source` | `videos/car-detection.mp4` | Video file, RTSP URL, or webcam index (`0`) |
| `camera_id` | `gate-1` | Name stored with each record |
| `loop_video` | `false` | Replay a video file endlessly (testing) |
| `process_every_n_frames` | `2` | Process every Nth frame. Raise it on slow PCs |
| `vehicle_model` | `yolov8n.pt` | Vehicle detector. `yolov8s.pt` is more accurate but slower |
| `vehicle_conf` | `0.4` | Minimum vehicle detection confidence |
| `imgsz` | `640` | YOLO input size |
| `device` | `cpu` | `cpu`, `0` (NVIDIA GPU), or `mps` (Apple) |
| `plate_model` | `models/plate.pt` | Plate detector weights |
| `ocr_languages` | `["en"]` | EasyOCR languages |
| `ocr_gpu` | `false` | Run OCR on GPU |
| `ocr_every_n_frames` | `2` | Read the plate every N processed frames per vehicle |
| `ocr_max_attempts` | `12` | Maximum OCR tries per vehicle |
| `min_vehicle_height_px` | `80` | Skip vehicles smaller than this |
| `min_plate_width_px` | `40` | Skip plate crops narrower than this |
| `plate_min_conf` | `0.30` | Discard single reads below this confidence |
| `plate_patterns` | `[A-Z]{2,4}\d{2,6}` | Valid plate formats (regex on letters and digits only) |
| `strict_format` | `false` | `true` discards reads that match no pattern |
| `confirm_reads` | `3` | Same plate read this many times means save now |
| `confirm_conf` | `0.60` | Average confidence needed for an early save |
| `min_reads_accept` | `2` | Agreeing reads needed for `confirmed` status |
| `high_conf_accept` | `0.80` | A single read this confident also counts as `confirmed` |
| `lost_frames` | `30` | Frames without the vehicle before its record is finalized |
| `min_track_frames` | `8` | Ignore vehicles seen for fewer frames |
| `cooldown_sec` | `60` | Same plate within this time is one record |
| `log_unreadable` | `true` | Save vehicles whose plate could not be read |
| `line` | `[[0.05,0.60],[0.95,0.60]]` | Entry line as two points (0 to 1). `null` turns it off |
| `entry_direction` | `positive` | Use `negative` if entry and exit look swapped |
| `db_path` | `data/plates.db` | Database file |
| `snapshot_dir` | `data/snapshots` | Image folder |

To support another plate style, add a regex to `plate_patterns`. For example, a Dutch-style `K884RS` is `[A-Z]\d{3}[A-Z]{2}`.

---

## 8. Tuning

| Problem | What to try |
|---|---|
| Plates are never read | Add or train `plate_model`. Lower `min_vehicle_height_px` or `min_plate_width_px`. Move the camera closer |
| Wrong plate text | Raise `confirm_reads` and `plate_min_conf`. Set `strict_format: true`. Fine-tune the plate detector |
| Duplicate records for one car | Raise `cooldown_sec` and `lost_frames` |
| Too slow | Raise `process_every_n_frames` or `ocr_every_n_frames`. Use a GPU. Use `yolov8n.pt` |
| Entry and exit swapped | Set `entry_direction: "negative"` |
| Different plate style | Edit `plate_patterns` |

**Camera tips.** The plate should be at least 100 px wide in the image. Use a slight downward angle at a distance of about 5 to 10 m,
good lighting, and an IR or low-light camera at night. Camera placement matters more than any model setting.

---

## 9. Test videos

Included or downloadable for testing:
- `car-detection.mp4`, `person-bicycle-car-detection.mp4`: wide traffic scenes from the Intel sample-videos repository (good for vehicle counting, plates are too small to read). Get them with `python scripts/download_samples.py`.
- `pk_plates_1.mp4`, `pk_plates_2.mp4`: handheld close-range clips with Pakistani plates (from the `ahmedhsn710/ANPR` repository, MIT license).
- `nl_plate_closeup.mp4`: close-range clip with a Dutch plate (from the `SpeedRadar-ANPR` repository).

The handheld clips are not fixed CCTV views, so set `line: null` when using them. Other free sources: the AGH Parking database
(videos with ground-truth plates) and free stock sites such as Mixkit, Pexels and Pixabay. Check each licence. These are for testing only.

---

## 10. Limitations

- Accuracy depends mostly on the plate detector and on the camera view. Without a plate model trained on your plate style, expect errors.
- Far-away or very small plates are skipped on purpose, because reading them would produce wrong text.
- Night, rain, motion blur and dirty or bent plates reduce accuracy. Use a suitable camera and lighting.
- Timestamps use the computer's clock. When you process a recorded video faster or slower than real time, `cooldown_sec` is still measured in real time, so use `video_sec` to see where in the video a record came from.
- The code is tested for its logic (voting, de-duplication, database, line crossing). Run it on your own camera footage and tune the settings before relying on it.

---

## 11. Privacy and legal

Number plates are personal data in many places. Get permission from the owner of the premises, tell people that the area is recorded
where required, keep the database and snapshots secure, and delete data you no longer need. Follow your local law.
