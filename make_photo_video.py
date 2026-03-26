#!/usr/bin/env python3
"""Create a single Full HD MP4 slideshow from baby/adult image pairs."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Styling constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.7
FONT_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
STROKE_COLOR = (0, 0, 0)
SHADOW_COLOR = (20, 20, 20)
SAFE_BOTTOM_MARGIN = 70
SHADOW_OFFSET = (3, 3)

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
    baby_path: Path
    adult_path: Path
    baby_warning: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Full HD MP4 video from baby/adult photo pairs listed in index.txt"
    )
    parser.add_argument("--images", required=True, type=Path, help="Folder containing images")
    parser.add_argument("--index", required=True, type=Path, help="Index file linking name -> adult ID")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    """Normalize names/stems for tolerant matching."""
    cleaned = value.lower().strip()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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

            parts = INDEX_SEP_PATTERN.split(line, maxsplit=1)
            if len(parts) < 3:
                parse_errors.append(
                    f"Line {line_no}: unable to parse row (no supported separator found): {line!r}"
                )
                continue

            name = parts[0].strip()
            adult_id = parts[2].strip()

            if not name:
                parse_errors.append(f"Line {line_no}: empty name")
                continue
            if not adult_id:
                parse_errors.append(f"Line {line_no}: empty adult ID")
                continue

            rows.append(IndexRow(line_no=line_no, name=name, adult_id=adult_id))
            name_counter[normalize_name(name)] += 1
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


def discover_images(images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str], List[Tuple[Path, str]]]:
    """Return adult dictionary, adult duplicate warnings, and baby candidate list.

    adult_map keys: numeric stem
    baby_candidates entries: (path, normalized_stem)
    """
    adult_map: Dict[str, Path] = {}
    adult_duplicate_warnings: Dict[str, str] = {}
    baby_candidates: List[Tuple[Path, str]] = []

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
            baby_candidates.append((path, normalize_name(stem)))

    return adult_map, adult_duplicate_warnings, baby_candidates


def choose_best_baby_match(name: str, candidates: Sequence[Tuple[Path, str]]) -> Tuple[Optional[Path], Optional[str]]:
    normalized_name = normalize_name(name)
    name_tokens = tokenize(name)
    scored: List[Tuple[Tuple[int, int, str], Path]] = []

    for path, normalized_stem in candidates:
        if not normalized_stem:
            continue

        contains_full = normalized_name in normalized_stem
        token_hits = 0
        if name_tokens:
            stem_tokens = set(normalized_stem.split())
            token_hits = sum(1 for t in name_tokens if t in stem_tokens)

        if not contains_full and token_hits < len(name_tokens):
            continue

        # Lower score tuple is better.
        # Priority: full-name containment, fewer extra chars, deterministic filename tie-break.
        extra_chars = max(len(normalized_stem) - len(normalized_name), 0)
        rank = 0 if contains_full else 1
        score = (rank, extra_chars, str(path).lower())
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


def prepare_1080p_frame(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: {img_path}")

    # Background: cover frame then blur.
    scale_bg = max(WIDTH / w, HEIGHT / h)
    bg_w, bg_h = max(1, int(round(w * scale_bg))), max(1, int(round(h * scale_bg)))
    bg = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    x0 = (bg_w - WIDTH) // 2
    y0 = (bg_h - HEIGHT) // 2
    bg = bg[y0 : y0 + HEIGHT, x0 : x0 + WIDTH]
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=20, sigmaY=20)

    # Foreground: contain in frame (no distortion).
    scale_fg = min(WIDTH / w, HEIGHT / h)
    fg_w, fg_h = max(1, int(round(w * scale_fg))), max(1, int(round(h * scale_fg)))
    fg = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_AREA if scale_fg < 1 else cv2.INTER_LINEAR)

    out = bg.copy()
    x1 = (WIDTH - fg_w) // 2
    y1 = (HEIGHT - fg_h) // 2
    out[y1 : y1 + fg_h, x1 : x1 + fg_w] = fg

    # Subtle border to separate foreground from blur.
    cv2.rectangle(out, (x1, y1), (x1 + fg_w - 1, y1 + fg_h - 1), (255, 255, 255), 1)
    return out


def draw_name_text(frame: np.ndarray, name: str, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return frame

    overlay = frame.copy()
    (text_w, text_h), baseline = cv2.getTextSize(name, FONT, FONT_SCALE, FONT_THICKNESS)

    x = max((WIDTH - text_w) // 2, 20)
    y = HEIGHT - SAFE_BOTTOM_MARGIN

    # Shadow
    cv2.putText(
        overlay,
        name,
        (x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]),
        FONT,
        FONT_SCALE,
        SHADOW_COLOR,
        FONT_THICKNESS + 4,
        cv2.LINE_AA,
    )

    # Stroke (4 directions)
    for ox, oy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        cv2.putText(
            overlay,
            name,
            (x + ox, y + oy),
            FONT,
            FONT_SCALE,
            STROKE_COLOR,
            FONT_THICKNESS + 3,
            cv2.LINE_AA,
        )

    # Main text
    cv2.putText(
        overlay,
        name,
        (x, y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

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
    baby_candidates: Sequence[Tuple[Path, str]],
) -> Tuple[List[MatchResult], List[str], List[str]]:
    matches: List[MatchResult] = []
    skipped: List[str] = []
    warnings: List[str] = []

    for row in rows:
        if not row.adult_id.isdigit():
            skipped.append(
                f"Line {row.line_no} '{row.name}': adult ID '{row.adult_id}' is not numeric"
            )
            continue

        adult_path = adult_map.get(row.adult_id)
        if adult_path is None:
            skipped.append(
                f"Line {row.line_no} '{row.name}': missing adult image for ID '{row.adult_id}'"
            )
            continue

        baby_path, baby_warning = choose_best_baby_match(row.name, baby_candidates)
        if baby_path is None:
            skipped.append(
                f"Line {row.line_no} '{row.name}': no matching baby image found"
            )
            continue

        if not baby_path.exists():
            skipped.append(f"Line {row.line_no} '{row.name}': baby file disappeared: {baby_path}")
            continue

        if not adult_path.exists():
            skipped.append(f"Line {row.line_no} '{row.name}': adult file disappeared: {adult_path}")
            continue

        match = MatchResult(
            row=row,
            baby_path=baby_path,
            adult_path=adult_path,
            baby_warning=baby_warning,
        )
        matches.append(match)
        if baby_warning:
            warnings.append(baby_warning)

    return matches, skipped, warnings


def write_segment(writer: cv2.VideoWriter, match: MatchResult) -> None:
    baby_frame = prepare_1080p_frame(match.baby_path)
    adult_frame = prepare_1080p_frame(match.adult_path)

    baby_frames = int(round(BABY_DURATION * FPS))
    fade_frames = int(round(FADE_DURATION * FPS))
    adult_frames = int(round(ADULT_DURATION * FPS))

    # 1) Baby-only section (no text)
    for _ in range(baby_frames):
        writer.write(baby_frame)

    # 2) Crossfade with text fade-in
    if fade_frames <= 1:
        blended = adult_frame.copy()
        with_text = draw_name_text(blended, match.row.name, alpha=1.0)
        writer.write(with_text)
    else:
        for i in range(fade_frames):
            alpha = i / (fade_frames - 1)
            blended = cv2.addWeighted(baby_frame, 1.0 - alpha, adult_frame, alpha, 0)
            with_text = draw_name_text(blended, match.row.name, alpha=alpha)
            writer.write(with_text)

    # 3) Adult-only section (text fully visible)
    adult_with_text = draw_name_text(adult_frame, match.row.name, alpha=1.0)
    for _ in range(adult_frames):
        writer.write(adult_with_text)


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
    try:
        iterator = matches
        if tqdm is not None:
            iterator = tqdm(matches, desc="Rendering", unit="person")

        for match in iterator:
            write_segment(writer, match)
    finally:
        writer.release()

    print_summary(
        total_rows=len(rows),
        parse_errors=parse_errors,
        matches=matches,
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
