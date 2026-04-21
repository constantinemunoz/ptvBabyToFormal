# Photo Pair Video Generator

Create one final **1920x1080 MP4** from many photos using an `index.txt` file that maps each person's name to their adult numeric photo ID.

The program renders each person as:

1. Baby photo only
2. Crossfade baby -> adult
3. Adult photo only

Default timing is `2s + 1s + 2s` (5 seconds per person), configurable in the script constants.

---

## 1) Prerequisites

- Python 3.10+ (3.11 recommended)
- A folder of `.jpg`, `.jpeg`, `.png` files
- `index.txt` mapping `Name -> numeric adult ID`

---

## 2) Install

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Prepare input files

Example structure:

```text
.
├── make_photo_video.py
├── requirements.txt
├── index.txt
└── images/
    ├── John Smith baby.jpg
    ├── John Smith scan 2.png
    ├── 101.jpg
    ├── Maria Lopez childhood photo.jpeg
    ├── 8473629105.png
    └── ...
```

### `index.txt` format

Any of the following separators are supported: `,` tab `|` `:` `=`

```text
# comments are allowed
John Smith,101
Maria Lopez | 8473629105
Chris Doe: 00048291
```

The final video order follows the same row order in `index.txt`.

For spreadsheet-like rows with many columns, the parser will:

- scan all columns and take the **first numeric adult ID** it finds
- also accept filename/path-style ID fields like `005606649.jpg` (ID extracted as `005606649`)
- auto-pick the most likely text/name columns (including common `LastName FirstName` trailing columns)
- convert common last-name-first patterns to on-screen `FirstName LastName`

---

## 4) Run

```bash
python make_photo_video.py --images ./images --index ./index.txt --output ./final_video.mp4
```

Or just run with defaults (uses `./images`, `./index.txt`, and writes `./final_video.mp4`):

```bash
python make_photo_video.py
```

Optional: specify MediaPipe Face Landmarker model file (recommended):

```bash
python make_photo_video.py --landmarker-model ./face_landmarker.task
```

---

## 5) What matching logic does

- **Adult photos**: matched by **numeric filename stem** exactly (`101.jpg`, `00048291.png`, etc.)
- **Baby photos**: matched flexibly from filename stem using normalized name matching:
  - case-insensitive
  - underscores/hyphens treated like spaces
  - punctuation mostly ignored
  - extra words allowed before/after the name
  - handles no-space name variants like `JohnSmith_baby.jpg`
  - handles reversed order variants like `Smith John childhood.png`

If multiple baby candidates match, the script picks the best deterministically and prints a warning.

---

## 6) Output and behavior

- Output: one MP4, 1920x1080, no audio
- Images are not stretched (aspect ratio preserved)
- Baby photo is transformed (scale/rotate/shift) to align eye positions with the matched adult photo using MediaPipe Face Landmarker eye points (with OpenCV fallbacks); adult photo is not modified
- Baby photo is center-cropped to the same aspect ratio as its matched adult photo before alignment (no squeeze/stretch)
- Frame style: centered foreground image over looping `background.mov` from the images folder (falls back if missing)
- Name text uses Arial Regular (when available) on a semi-transparent black banner with yellow border (`#FCB315`) at the bottom, starting when the baby->adult fade begins
- For entries missing a baby photo match, the person is still rendered with an adult-only segment (3 seconds, no crossfade), and a warning is included in the summary
- If `titleCard.mp4` exists in the images folder, it plays once at the beginning and then loops 3 times at the end
- Adds an additional 10-frame inter-segment cross dissolve from each adult photo to the next person's baby photo
- Uses a fixed-size black banner with a single yellow top line (no per-name box resizing), centered in the same position each time
- Uses one uniform font size for all names, chosen so the longest name fits the frame width

---

## 7) Adjust timings and output

Edit constants at top of `make_photo_video.py`:

- `BABY_DURATION`
- `FADE_DURATION`
- `ADULT_DURATION`
- `FPS`
- `WIDTH`
- `HEIGHT`

---

## 8) Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
Install dependencies in the active virtual environment:

```bash
pip install -r requirements.txt
```

### "No valid pairs to render"
Check that:

- Each `index.txt` adult ID is numeric
- Adult image stems exist exactly as listed IDs
- Baby filenames contain names (normalized/flexible matching)
- Files are `.jpg/.jpeg/.png`

The summary now reports detailed per-person skip reasons, including combinations such as:
- adult missing + baby missing
- adult unreadable + baby readable
- baby unreadable + adult readable
- both unreadable

### MP4 not playing on one device
The script attempts codec tags in order: `avc1`, `H264`, then `mp4v`.
If H.264 encoder is unavailable in your OpenCV build, it may fall back to `mp4v`.

### Eye alignment did not happen for some people
If eye pairs cannot be detected reliably, the script infers eye points from face boxes and still performs alignment. If face detection is also unavailable, it falls back to center-crop + resize (still preserving aspect ratio and no stretching) and logs a warning in the summary.

### Background video not visible
Place `background.mov` inside your `--images` folder. The script loops it continuously until rendering finishes. If missing/unreadable, it falls back to a solid background and prints a warning.

---

## 9) Example command for larger runs

```bash
python make_photo_video.py \
  --images /data/photos \
  --index /data/index.txt \
  --output /data/final_video.mp4
```
