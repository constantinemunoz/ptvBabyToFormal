#!/usr/bin/env python3
"""Create a single Full HD MP4 slideshow from baby/adult image pairs."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    tqdm = None


# =========================
# Easy-to-edit timing/video constants
# =========================
BABY_DURATION = 2.0
FADE_DURATION = 1.0
ADULT_DURATION = 2.0
MISSING_BABY_ADULT_ONLY_DURATION = 3.0
FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Styling constants
# Arial Regular text settings
FONT_SIZE = 74
FONT_STROKE_WIDTH = 4
TEXT_COLOR = (255, 255, 255)
STROKE_COLOR = (0, 0, 0)
SHADOW_COLOR = (20, 20, 20)
SAFE_BOTTOM_MARGIN = 70
SHADOW_OFFSET = (3, 3)
BANNER_FILL_COLOR = (0, 0, 0)  # black
BANNER_BORDER_COLOR = (21, 179, 252)  # #FCB315 in BGR
BANNER_PADDING_X = 34
BANNER_PADDING_Y = 20
BANNER_BORDER_THICKNESS = 3
BANNER_ALPHA = 0.55
GREEN_BG_COLOR = (0, 255, 0)  # pure green background for chroma keying
MIN_EYE_DIST_RATIO = 0.12
MAX_EYE_DIST_RATIO = 0.75
MAX_EYE_LINE_HEIGHT_RATIO = 0.72
MIN_ALIGN_SCALE = 0.75
MAX_ALIGN_SCALE = 1.25
MIN_FACE_ALIGN_SCALE = 0.80
MAX_FACE_ALIGN_SCALE = 1.20
MAX_ALIGNMENT_ANGLE_DEG = 15.0

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INDEX_SEP_PATTERN = re.compile(r"\s*(,|\t|\||:|=)\s*")


@dataclass
class IndexRow:
    line_no: int
    name: str
    adult_id: str


@dataclass
class MatchResult:
    row: IndexRow
    baby_path: Optional[Path]
    adult_path: Path
    baby_warning: Optional[str] = None
    missing_baby: bool = False


@dataclass(frozen=True)
class BabyCandidate:
    path: Path
    normalized_stem: str
    compact_stem: str


@dataclass
class ContainedForeground:
    img: np.ndarray
    x: int
    y: int


class LoopingBackground:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.cap = None
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                self.cap = cap

    def next_frame(self) -> np.ndarray:
        if self.cap is None:
            return np.full((self.height, self.width, 3), GREEN_BG_COLOR, dtype=np.uint8)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return np.full((self.height, self.width, 3), GREEN_BG_COLOR, dtype=np.uint8)
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class MediaPipeEyeDetector:
    LEFT_EYE_IDX = 33
    RIGHT_EYE_IDX = 263

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.landmarker = None
        try:
            import mediapipe as mp  # type: ignore

            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
            )
            self.landmarker = FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception:
            self.landmarker = None
            self._mp = None

    def detect_eye_centers(self, bgr_img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.landmarker is None or self._mp is None:
            return None
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        h, w = bgr_img.shape[:2]
        left_lm = landmarks[self.LEFT_EYE_IDX]
        right_lm = landmarks[self.RIGHT_EYE_IDX]
        left = np.array([left_lm.x * w, left_lm.y * h], dtype=np.float32)
        right = np.array([right_lm.x * w, right_lm.y * h], dtype=np.float32)
        if left[0] > right[0]:
            # mirrored/unreliable output; caller should fallback.
            return None
        return left, right


def load_arial_font(size: int) -> ImageFont.ImageFont:
    # Common Arial locations across platforms.
    candidates = [
        "arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/microsoft/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    # Fallback if Arial is unavailable in runtime environment.
    return ImageFont.load_default()


def load_eye_detector() -> Optional[cv2.CascadeClassifier]:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


def load_face_detector() -> Optional[cv2.CascadeClassifier]:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


def extract_numeric_id(token: str) -> Optional[str]:
    """Extract a numeric ID from a token or filename/path token."""
    candidate = token.strip()
    if not candidate:
        return None
    if candidate.isdigit():
        return candidate

    # Accept values such as "005606649.jpg" or "images/005606649.png".
    stem = Path(candidate).stem.strip()
    if stem.isdigit():
        return stem
    return None


def choose_name_from_tokens(tokens: Sequence[str], adult_id_token_index: Optional[int]) -> Optional[str]:
    """Pick a likely person-name field from a tokenized row."""
    name_candidates: List[Tuple[int, int, str]] = []

    for i, token in enumerate(tokens):
        if adult_id_token_index is not None and i == adult_id_token_index:
            continue

        t = token.strip()
        if not t:
            continue

        # Skip obvious non-name fields.
        if extract_numeric_id(t) is not None:
            continue
        if Path(t).suffix.lower() in SUPPORTED_EXTENSIONS:
            continue
        if "/" in t or "\\" in t:
            continue

        alpha_chars = sum(ch.isalpha() for ch in t)
        if alpha_chars == 0:
            continue

        # Score: higher alphabetic count and later columns (often last/first name fields).
        name_candidates.append((alpha_chars, i, t))

    if not name_candidates:
        return None

    # Prefer the strongest textual candidate; break ties by later column index.
    name_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_single = name_candidates[0]

    # If the last two usable columns are textual, combine them as "First Last".
    usable = sorted(name_candidates, key=lambda x: x[1])
    if len(usable) >= 2:
        last = usable[-1]
        prev = usable[-2]
        if last[1] == prev[1] + 1:
            combined = f"{last[2]} {prev[2]}".strip()
            if len(combined) >= len(best_single[2]):
                return combined

    return best_single[2]


def normalize_display_name(name: str) -> str:
    """Convert common last-name-first formats into first-name-last for on-screen display/matching."""
    raw = name.strip()
    if not raw:
        return raw

    # "Last, First [Middle]" -> "First [Middle] Last"
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 2:
            return " ".join(parts[1:] + [parts[0]])

    return raw


def crop_to_aspect(img: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    src_aspect = w / h
    if abs(src_aspect - target_aspect) < 1e-6:
        return img

    if src_aspect > target_aspect:
        # too wide: crop width
        new_w = int(round(h * target_aspect))
        x0 = max((w - new_w) // 2, 0)
        return img[:, x0 : x0 + new_w]

    # too tall: crop height
    new_h = int(round(w / target_aspect))
    y0 = max((h - new_h) // 2, 0)
    return img[y0 : y0 + new_h, :]


def detect_primary_face(
    img: np.ndarray,
    face_detector: Optional[cv2.CascadeClassifier],
) -> Optional[Tuple[int, int, int, int]]:
    if face_detector is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def infer_eye_centers_from_face(face_box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate eye centers from a face rectangle when explicit eye detection fails."""
    x, y, w, h = face_box
    left = np.array([x + 0.32 * w, y + 0.40 * h], dtype=np.float32)
    right = np.array([x + 0.68 * w, y + 0.40 * h], dtype=np.float32)
    return left, right


def detect_eye_centers(
    img: np.ndarray,
    eye_detector: cv2.CascadeClassifier,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if face_box is not None:
        fx, fy, fw, fh = face_box
        y0 = fy
        y1 = fy + int(fh * 0.70)  # eyes are typically in upper face area
        x0 = fx
        x1 = fx + fw
        roi = gray[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)]
        if roi.size == 0:
            return None
        eyes = eye_detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(14, 14))
        eyes = [(ex + x0, ey + y0, ew, eh) for (ex, ey, ew, eh) in eyes]
    else:
        eyes = eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
    if len(eyes) < 2:
        return None

    # Pick two largest detections; then sort left-to-right.
    eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:6]
    h, w = img.shape[:2]
    best_pair = None
    best_score = float("inf")
    for i in range(len(eyes_sorted)):
        for j in range(i + 1, len(eyes_sorted)):
            ex1, ey1, ew1, eh1 = eyes_sorted[i]
            ex2, ey2, ew2, eh2 = eyes_sorted[j]
            c1 = np.array([ex1 + ew1 / 2.0, ey1 + eh1 / 2.0], dtype=np.float32)
            c2 = np.array([ex2 + ew2 / 2.0, ey2 + eh2 / 2.0], dtype=np.float32)
            if abs(c1[0] - c2[0]) < 10:
                continue
            inter_eye_dist = float(np.linalg.norm(c2 - c1))
            dist_ratio = inter_eye_dist / max(w, 1)
            avg_y_ratio = ((c1[1] + c2[1]) * 0.5) / max(h, 1)
            # Reject implausible detections that often cause bad zoom or drift.
            if dist_ratio < MIN_EYE_DIST_RATIO or dist_ratio > MAX_EYE_DIST_RATIO:
                continue
            if avg_y_ratio > MAX_EYE_LINE_HEIGHT_RATIO:
                continue
            # Penalize large vertical gap to prefer natural eye-line.
            score = abs(c1[1] - c2[1]) + (avg_y_ratio * 10.0)
            if score < best_score:
                best_score = score
                best_pair = (c1, c2)

    if best_pair is None:
        return None

    left, right = sorted(best_pair, key=lambda p: p[0])
    return left, right


def align_baby_to_adult(
    baby_img: np.ndarray,
    adult_img: np.ndarray,
    mp_detector: Optional[MediaPipeEyeDetector],
    eye_detector: Optional[cv2.CascadeClassifier],
    face_detector: Optional[cv2.CascadeClassifier],
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Align baby photo to adult eye position via similarity transform.
    Adult image is never modified.
    """
    target_h, target_w = adult_img.shape[:2]
    target_aspect = target_w / max(target_h, 1)
    baby_cropped = crop_to_aspect(baby_img, target_aspect)

    if eye_detector is None and face_detector is None:
        resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, "Eye/face detector unavailable; used center crop fallback."

    baby_face = detect_primary_face(baby_cropped, face_detector)
    adult_face = detect_primary_face(adult_img, face_detector)

    baby_eyes = mp_detector.detect_eye_centers(baby_cropped) if mp_detector else None
    adult_eyes = mp_detector.detect_eye_centers(adult_img) if mp_detector else None
    source = "mediapipe"
    if baby_eyes is None or adult_eyes is None:
        baby_eyes = detect_eye_centers(baby_cropped, eye_detector, baby_face) if eye_detector else None
        adult_eyes = detect_eye_centers(adult_img, eye_detector, adult_face) if eye_detector else None
        source = "cascade"
    baby_inferred = False
    adult_inferred = False

    if baby_eyes is None and baby_face is not None:
        baby_eyes = infer_eye_centers_from_face(baby_face)
        baby_inferred = True
    if adult_eyes is None and adult_face is not None:
        adult_eyes = infer_eye_centers_from_face(adult_face)
        adult_inferred = True

    if baby_eyes and adult_eyes:
        b_left, b_right = baby_eyes
        a_left, a_right = adult_eyes

        b_vec = b_right - b_left
        a_vec = a_right - a_left
        b_dist = float(np.linalg.norm(b_vec))
        a_dist = float(np.linalg.norm(a_vec))
        if b_dist < 1e-6 or a_dist < 1e-6:
            resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return resized, "Invalid eye geometry; used center crop fallback."

        scale = a_dist / b_dist
        if scale < MIN_ALIGN_SCALE or scale > MAX_ALIGN_SCALE:
            resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return (
                resized,
                f"Eye-based scale {scale:.2f} outside safe range [{MIN_ALIGN_SCALE:.2f}, {MAX_ALIGN_SCALE:.2f}]; used center crop fallback.",
            )
        b_angle = math.atan2(float(b_vec[1]), float(b_vec[0]))
        a_angle = math.atan2(float(a_vec[1]), float(a_vec[0]))
        if baby_inferred or adult_inferred:
            # Inferred eyes are horizontal by construction; avoid introducing unstable rotation.
            b_angle = 0.0
            a_angle = 0.0
        theta = a_angle - b_angle
        if abs(math.degrees(theta)) > MAX_ALIGNMENT_ANGLE_DEG:
            resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return resized, (
                f"Eye-based rotation {math.degrees(theta):.1f}° too large; skipped eye alignment to avoid mirrored/unnatural output."
            )
        cos_t = math.cos(theta) * scale
        sin_t = math.sin(theta) * scale
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        trans = a_left - rot @ b_left
        matrix = np.array(
            [[rot[0, 0], rot[0, 1], trans[0]], [rot[1, 0], rot[1, 1], trans[1]]],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            baby_cropped,
            matrix,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if baby_inferred or adult_inferred:
            return aligned, "Used inferred eye points from face boxes (explicit eye detection unavailable)."
        return aligned, f"Aligned using {source} eye detection."

    # Face-based fallback (more stable than random eye detections).
    if baby_face is not None and adult_face is not None:
        bx, by, bw, bh = baby_face
        ax, ay, aw, ah = adult_face
        b_center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float32)
        a_center = np.array([ax + aw / 2.0, ay + ah / 2.0], dtype=np.float32)
        scale = (aw / max(bw, 1))
        scale = float(np.clip(scale, MIN_FACE_ALIGN_SCALE, MAX_FACE_ALIGN_SCALE))
        matrix = np.array(
            [[scale, 0.0, a_center[0] - scale * b_center[0]], [0.0, scale, a_center[1] - scale * b_center[1]]],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            baby_cropped,
            matrix,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned, "Used face-based alignment fallback (eye pair not reliable)."

    # Last-resort fallback.
    if not baby_eyes or not adult_eyes:
        resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, "Eye/face detection failed for baby/adult; used center crop fallback."
    resized = cv2.resize(baby_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, "Used center crop fallback."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Full HD MP4 video from baby/adult photo pairs listed in index.txt"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("./images"),
        help="Folder containing images (default: ./images)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("./index.txt"),
        help="Index file linking name -> adult ID (default: ./index.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./final_video.mp4"),
        help="Output MP4 path (default: ./final_video.mp4)",
    )
    parser.add_argument(
        "--landmarker-model",
        type=Path,
        default=Path("./face_landmarker.task"),
        help="MediaPipe Face Landmarker model path (default: ./face_landmarker.task)",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    """Normalize names/stems for tolerant matching."""
    cleaned = value.lower().strip()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compact_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize(value: str) -> List[str]:
    return [token for token in normalize_name(value).split(" ") if token]


def parse_index_file(index_path: Path) -> Tuple[List[IndexRow], List[str], List[str], List[str]]:
    rows: List[IndexRow] = []
    parse_errors: List[str] = []
    duplicate_name_warnings: List[str] = []
    duplicate_id_warnings: List[str] = []

    name_counter: Counter[str] = Counter()
    id_counter: Counter[str] = Counter()

    with index_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Keep all columns (not only first split) so we can handle spreadsheet-like rows.
            parts = [p.strip() for p in INDEX_SEP_PATTERN.split(line) if p.strip() and p not in {",", "\t", "|", ":", "="}]
            if len(parts) < 2:
                parse_errors.append(f"Line {line_no}: unable to parse row: {line!r}")
                continue

            adult_id = None
            adult_id_token_index: Optional[int] = None
            for i, part in enumerate(parts):
                maybe_id = extract_numeric_id(part)
                if maybe_id is not None:
                    adult_id = maybe_id
                    adult_id_token_index = i
                    break

            name = choose_name_from_tokens(parts, adult_id_token_index)

            if adult_id is None:
                parse_errors.append(f"Line {line_no}: no numeric adult ID found in row")
                continue

            if not name:
                parse_errors.append(f"Line {line_no}: empty name")
                continue

            display_name = normalize_display_name(name)

            rows.append(IndexRow(line_no=line_no, name=display_name, adult_id=adult_id))
            name_counter[normalize_name(display_name)] += 1
            id_counter[adult_id] += 1

    for normalized_name, count in name_counter.items():
        if count > 1:
            duplicate_name_warnings.append(
                f"Duplicate name in index ({count} rows): '{normalized_name}'"
            )

    for adult_id, count in id_counter.items():
        if count > 1:
            duplicate_id_warnings.append(
                f"Duplicate adult ID in index ({count} rows): '{adult_id}'"
            )

    return rows, parse_errors, duplicate_name_warnings, duplicate_id_warnings


def discover_images(images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str], List[BabyCandidate]]:
    """Return adult dictionary, adult duplicate warnings, and baby candidate list.

    adult_map keys: numeric stem
    baby_candidates entries: BabyCandidate(path, normalized_stem, compact_stem)
    """
    adult_map: Dict[str, Path] = {}
    adult_duplicate_warnings: Dict[str, str] = {}
    baby_candidates: List[BabyCandidate] = []

    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stem = path.stem.strip()
        if stem.isdigit():
            if stem in adult_map:
                existing = adult_map[stem]
                chosen = min(existing, path, key=lambda p: str(p).lower())
                adult_map[stem] = chosen
                adult_duplicate_warnings[stem] = (
                    f"Duplicate adult image stem '{stem}': keeping '{chosen.name}', skipping another match."
                )
            else:
                adult_map[stem] = path
        else:
            norm_stem = normalize_name(stem)
            baby_candidates.append(
                BabyCandidate(path=path, normalized_stem=norm_stem, compact_stem=compact_alnum(stem))
            )

    return adult_map, adult_duplicate_warnings, baby_candidates


def choose_best_baby_match(name: str, candidates: Sequence[BabyCandidate]) -> Tuple[Optional[Path], Optional[str]]:
    normalized_name = normalize_name(name)
    name_tokens = tokenize(name)
    name_compact = compact_alnum(name)
    reversed_name = " ".join(reversed(name_tokens)).strip()
    reversed_compact = compact_alnum(reversed_name)
    scored: List[Tuple[Tuple[int, int, int, int, str], Path]] = []

    for candidate in candidates:
        path = candidate.path
        normalized_stem = candidate.normalized_stem
        compact_stem = candidate.compact_stem
        if not normalized_stem:
            continue

        contains_full = normalized_name in normalized_stem
        contains_reversed = bool(reversed_name) and reversed_name in normalized_stem
        contains_compact = bool(name_compact) and name_compact in compact_stem
        contains_reversed_compact = bool(reversed_compact) and reversed_compact in compact_stem

        token_hits = 0
        if name_tokens:
            stem_tokens = set(normalized_stem.split())
            token_hits = sum(1 for t in name_tokens if t in stem_tokens)

        if (
            not contains_full
            and not contains_reversed
            and not contains_compact
            and not contains_reversed_compact
            and token_hits < len(name_tokens)
        ):
            continue

        # Lower score tuple is better.
        # Priority:
        # 1) direct full-name containment,
        # 2) compact full-name containment (handles no-space filenames),
        # 3) reversed-name containment (last-first variants),
        # 4) fewer extra chars,
        # 5) deterministic path tie-break.
        extra_chars = max(len(normalized_stem) - len(normalized_name), 0)
        rank_direct = 0 if contains_full else 1
        rank_compact = 0 if contains_compact else 1
        rank_reversed = 0 if (contains_reversed or contains_reversed_compact) else 1
        score = (rank_direct, rank_compact, rank_reversed, extra_chars, str(path).lower())
        scored.append((score, path))

    if not scored:
        return None, None

    scored.sort(key=lambda item: item[0])
    best = scored[0][1]
    warning = None

    if len(scored) > 1:
        preview = ", ".join(p.name for _, p in scored[:3])
        warning = (
            f"Multiple baby matches for '{name}'. Selected '{best.name}'. Top candidates: {preview}"
        )

    return best, warning


def prepare_contained_foreground(img: np.ndarray) -> ContainedForeground:
    if img is None:
        raise ValueError("Cannot prepare foreground from empty image")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image dimensions")

    # Foreground: contain in frame (no distortion).
    scale_fg = min(WIDTH / w, HEIGHT / h)
    fg_w, fg_h = max(1, int(round(w * scale_fg))), max(1, int(round(h * scale_fg)))
    fg = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_AREA if scale_fg < 1 else cv2.INTER_LINEAR)

    x1 = (WIDTH - fg_w) // 2
    y1 = (HEIGHT - fg_h) // 2
    return ContainedForeground(img=fg, x=x1, y=y1)


def compose_with_background(background: np.ndarray, fg: ContainedForeground) -> np.ndarray:
    out = background.copy()
    fh, fw = fg.img.shape[:2]
    out[fg.y : fg.y + fh, fg.x : fg.x + fw] = fg.img
    return out


def prepare_1080p_frame(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    fg = prepare_contained_foreground(img)
    bg = np.full((HEIGHT, WIDTH, 3), GREEN_BG_COLOR, dtype=np.uint8)
    return compose_with_background(bg, fg)


def draw_name_text(frame: np.ndarray, name: str, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return frame

    overlay = frame.copy()
    font = load_arial_font(FONT_SIZE)
    pil_probe = Image.new("RGB", (WIDTH, HEIGHT))
    probe_draw = ImageDraw.Draw(pil_probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), name, font=font, stroke_width=FONT_STROKE_WIDTH)
    text_w = max(1, right - left)
    text_h = max(1, bottom - top)
    x = max((WIDTH - text_w) // 2, 20)
    banner_center_y = HEIGHT - SAFE_BOTTOM_MARGIN

    # Semi-transparent banner with yellow border behind text.
    top_left = (
        max(x - BANNER_PADDING_X, 10),
        max(int(banner_center_y - (text_h / 2) - BANNER_PADDING_Y), 10),
    )
    bottom_right = (
        min(x + text_w + BANNER_PADDING_X, WIDTH - 10),
        min(int(banner_center_y + (text_h / 2) + BANNER_PADDING_Y), HEIGHT - 10),
    )
    cv2.rectangle(overlay, top_left, bottom_right, BANNER_FILL_COLOR, thickness=-1)
    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        BANNER_BORDER_COLOR,
        thickness=BANNER_BORDER_THICKNESS,
    )

    banner_blended = cv2.addWeighted(overlay, BANNER_ALPHA, frame, 1.0 - BANNER_ALPHA, 0)
    overlay = banner_blended.copy()

    # Draw shadow + stroked text using PIL (Arial Regular when available).
    pil_overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_overlay)
    box_w = bottom_right[0] - top_left[0]
    box_h = bottom_right[1] - top_left[1]
    text_x = top_left[0] + (box_w - text_w) // 2 - left
    text_y = top_left[1] + (box_h - text_h) // 2 - top
    shadow_pos = (text_x + SHADOW_OFFSET[0], text_y + SHADOW_OFFSET[1])
    text_pos = (text_x, text_y)
    draw.text(
        shadow_pos,
        name,
        font=font,
        fill=SHADOW_COLOR[::-1],  # RGB
        stroke_width=FONT_STROKE_WIDTH + 1,
        stroke_fill=STROKE_COLOR[::-1],
    )
    draw.text(
        text_pos,
        name,
        font=font,
        fill=TEXT_COLOR[::-1],  # RGB
        stroke_width=FONT_STROKE_WIDTH,
        stroke_fill=STROKE_COLOR[::-1],
    )
    overlay = cv2.cvtColor(np.array(pil_overlay), cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def create_writer(output_path: Path) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fourcc_name in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(FPS),
            (WIDTH, HEIGHT),
            isColor=True,
        )
        if writer.isOpened():
            print(f"Using video codec tag: {fourcc_name}")
            return writer
        writer.release()

    raise RuntimeError("Unable to open video writer with avc1/H264/mp4v codecs")


def build_matches(
    rows: Sequence[IndexRow],
    adult_map: Dict[str, Path],
    baby_candidates: Sequence[BabyCandidate],
) -> Tuple[List[MatchResult], List[str], List[str]]:
    matches: List[MatchResult] = []
    skipped: List[str] = []
    warnings: List[str] = []

    for row in rows:
        row_reasons: List[str] = []
        adult_path: Optional[Path] = None
        baby_path, baby_warning = choose_best_baby_match(row.name, baby_candidates)
        baby_missing_reason: Optional[str] = None

        if not row.adult_id.isdigit():
            row_reasons.append(f"adult ID '{row.adult_id}' is not numeric")
        else:
            adult_path = adult_map.get(row.adult_id)
            if adult_path is None:
                row_reasons.append(f"adult photo missing for ID '{row.adult_id}'")
            elif not adult_path.exists() or not adult_path.is_file():
                row_reasons.append(f"adult photo path is missing on disk: {adult_path}")
            else:
                adult_probe = cv2.imread(str(adult_path), cv2.IMREAD_COLOR)
                if adult_probe is None:
                    row_reasons.append(f"adult photo is not readable: {adult_path}")

        if baby_path is None:
            baby_missing_reason = "baby photo missing (no filename match found)"
            row_reasons.append(baby_missing_reason)
        elif not baby_path.exists() or not baby_path.is_file():
            row_reasons.append(f"baby photo path is missing on disk: {baby_path}")
        else:
            baby_probe = cv2.imread(str(baby_path), cv2.IMREAD_COLOR)
            if baby_probe is None:
                row_reasons.append(f"baby photo is not readable: {baby_path}")

        only_missing_baby = (
            baby_missing_reason is not None
            and len(row_reasons) == 1
            and adult_path is not None
        )
        if row_reasons and not only_missing_baby:
            skipped.append(
                f"Line {row.line_no} '{row.name}': skipped because " + "; ".join(row_reasons)
            )
            continue

        assert adult_path is not None
        if only_missing_baby:
            warnings.append(
                f"Line {row.line_no} '{row.name}': baby photo missing; rendered adult-only segment ({MISSING_BABY_ADULT_ONLY_DURATION:.1f}s)."
            )
        match = MatchResult(
            row=row,
            baby_path=baby_path,
            adult_path=adult_path,
            baby_warning=baby_warning,
            missing_baby=only_missing_baby,
        )
        matches.append(match)
        if baby_warning:
            warnings.append(baby_warning)

    return matches, skipped, warnings


def write_segment(
    writer: cv2.VideoWriter,
    match: MatchResult,
    background: LoopingBackground,
    mp_detector: Optional[MediaPipeEyeDetector],
    eye_detector: Optional[cv2.CascadeClassifier],
    face_detector: Optional[cv2.CascadeClassifier],
    warnings_out: List[str],
) -> Optional[str]:
    adult_img = cv2.imread(str(match.adult_path), cv2.IMREAD_COLOR)
    if adult_img is None:
        return f"Line {match.row.line_no} '{match.row.name}': adult photo became unreadable at render time: {match.adult_path}"

    adult_fg = prepare_contained_foreground(adult_img)
    if match.missing_baby:
        for _ in range(int(round(MISSING_BABY_ADULT_ONLY_DURATION * FPS))):
            adult_frame = compose_with_background(background.next_frame(), adult_fg)
            adult_with_text = draw_name_text(adult_frame, match.row.name, alpha=1.0)
            writer.write(adult_with_text)
        return None

    if match.baby_path is None:
        return f"Line {match.row.line_no} '{match.row.name}': internal error: baby path missing for full segment."

    baby_img = cv2.imread(str(match.baby_path), cv2.IMREAD_COLOR)
    if baby_img is None:
        return f"Line {match.row.line_no} '{match.row.name}': baby photo became unreadable at render time: {match.baby_path}"

    aligned_baby_img, align_warning = align_baby_to_adult(
        baby_img, adult_img, mp_detector, eye_detector, face_detector
    )
    if align_warning:
        warnings_out.append(f"{match.row.name}: {align_warning}")

    baby_fg = prepare_contained_foreground(aligned_baby_img)

    baby_frames = int(round(BABY_DURATION * FPS))
    fade_frames = int(round(FADE_DURATION * FPS))
    adult_frames = int(round(ADULT_DURATION * FPS))

    # 1) Baby-only section (no text)
    for _ in range(baby_frames):
        baby_frame = compose_with_background(background.next_frame(), baby_fg)
        writer.write(baby_frame)

    # 2) Crossfade with text fade-in
    if fade_frames <= 1:
        adult_frame = compose_with_background(background.next_frame(), adult_fg)
        blended = adult_frame.copy()
        with_text = draw_name_text(blended, match.row.name, alpha=1.0)
        writer.write(with_text)
    else:
        for i in range(fade_frames):
            alpha = i / (fade_frames - 1)
            bg = background.next_frame()
            baby_frame = compose_with_background(bg, baby_fg)
            adult_frame = compose_with_background(bg, adult_fg)
            blended = cv2.addWeighted(baby_frame, 1.0 - alpha, adult_frame, alpha, 0)
            with_text = draw_name_text(blended, match.row.name, alpha=alpha)
            writer.write(with_text)

    # 3) Adult-only section (text fully visible)
    for _ in range(adult_frames):
        adult_frame = compose_with_background(background.next_frame(), adult_fg)
        adult_with_text = draw_name_text(adult_frame, match.row.name, alpha=1.0)
        writer.write(adult_with_text)
    return None


def print_summary(
    total_rows: int,
    parse_errors: Sequence[str],
    matches: Sequence[MatchResult],
    skipped: Sequence[str],
    warnings: Sequence[str],
) -> None:
    print("\n===== SUMMARY =====")
    print(f"Total index entries read: {total_rows}")
    print(f"Matched pairs rendered:   {len(matches)}")
    print(f"Skipped entries:          {len(skipped)}")

    if parse_errors:
        print("\nIndex parse errors:")
        for err in parse_errors:
            print(f"  - {err}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if skipped:
        print("\nSkipped details:")
        for reason in skipped:
            print(f"  - {reason}")


def main() -> int:
    args = parse_args()

    if not args.images.exists() or not args.images.is_dir():
        print(f"ERROR: --images folder does not exist or is not a directory: {args.images}")
        return 1

    if not args.index.exists() or not args.index.is_file():
        print(f"ERROR: --index file does not exist: {args.index}")
        return 1

    rows, parse_errors, duplicate_name_warnings, duplicate_id_warnings = parse_index_file(args.index)
    adult_map, adult_dup_warnings, baby_candidates = discover_images(args.images)

    aggregate_warnings: List[str] = []
    aggregate_warnings.extend(duplicate_name_warnings)
    aggregate_warnings.extend(duplicate_id_warnings)
    aggregate_warnings.extend(adult_dup_warnings.values())

    matches, skipped, match_warnings = build_matches(rows, adult_map, baby_candidates)
    aggregate_warnings.extend(match_warnings)
    mp_detector: Optional[MediaPipeEyeDetector] = None
    if args.landmarker_model.exists():
        mp_detector = MediaPipeEyeDetector(args.landmarker_model)
        if mp_detector.landmarker is None:
            aggregate_warnings.append(
                f"MediaPipe Face Landmarker model found but failed to load: {args.landmarker_model}"
            )
    else:
        aggregate_warnings.append(
            f"MediaPipe Face Landmarker model not found at {args.landmarker_model}; using cascade/face fallback."
        )

    eye_detector = load_eye_detector()
    face_detector = load_face_detector()
    if eye_detector is None:
        aggregate_warnings.append("Could not load OpenCV eye detector.")
    if face_detector is None:
        aggregate_warnings.append("Could not load OpenCV face detector.")
    if eye_detector is None and face_detector is None:
        aggregate_warnings.append("Both eye and face detectors unavailable; using crop fallback for alignment.")

    if not matches:
        print_summary(
            total_rows=len(rows),
            parse_errors=parse_errors,
            matches=matches,
            skipped=skipped,
            warnings=aggregate_warnings,
        )
        print("\nNo valid pairs to render. Exiting without creating video.")
        return 2

    writer = create_writer(args.output)
    background = LoopingBackground(args.images / "background.mov", WIDTH, HEIGHT)
    if background.cap is None:
        aggregate_warnings.append(
            f"background.mov was not found/readable at {args.images / 'background.mov'}; using solid background fallback."
        )
    rendered_matches: List[MatchResult] = []
    try:
        iterator = matches
        if tqdm is not None:
            iterator = tqdm(matches, desc="Rendering", unit="person")

        for match in iterator:
            render_error = write_segment(
                writer, match, background, mp_detector, eye_detector, face_detector, aggregate_warnings
            )
            if render_error:
                skipped.append(render_error)
                continue
            rendered_matches.append(match)
    finally:
        background.release()
        writer.release()

    print_summary(
        total_rows=len(rows),
        parse_errors=parse_errors,
        matches=rendered_matches,
        skipped=skipped,
        warnings=aggregate_warnings,
    )

    segment_seconds = BABY_DURATION + FADE_DURATION + ADULT_DURATION
    print(f"\nWrote video: {args.output}")
    print(f"Segment duration per person: {segment_seconds:.3f}s")
    print(f"Output settings: {WIDTH}x{HEIGHT} @ {FPS} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
