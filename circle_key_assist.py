from __future__ import annotations

import argparse
import configparser
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mss
import numpy as np
from pynput import keyboard


@dataclass
class Detection:
    inside: bool
    digit: Optional[str]
    red_angle: Optional[float]
    target_start: Optional[float]
    target_end: Optional[float]
    digit_confidence: float
    red_pixels: int
    blue_pixels: int
    center_x: float
    center_y: float
    radius: float


@dataclass
class RuntimeState:
    active: bool = False
    quitting: bool = False


@dataclass
class PendingPress:
    digit: str
    press_at: float


def hotkey_name(name: str) -> str:
    lowered = name.strip().lower()
    if len(lowered) == 1:
        return lowered
    if lowered.startswith("<") and lowered.endswith(">"):
        return lowered
    return f"<{lowered}>"


def angle_clockwise_from_top(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> np.ndarray:
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    return (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0


def circular_mean_deg(angles: np.ndarray) -> Optional[float]:
    if angles.size == 0:
        return None
    radians = np.deg2rad(angles)
    sin_sum = float(np.sin(radians).sum())
    cos_sum = float(np.cos(radians).sum())
    if sin_sum == 0.0 and cos_sum == 0.0:
        return None
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def circular_arc_span(angles: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Return the smallest clockwise arc containing the angle cluster."""
    if angles.size == 0:
        return None, None
    sorted_angles = np.sort(angles.astype(np.float32))
    if sorted_angles.size == 1:
        value = float(sorted_angles[0])
        return value, value

    gaps = np.diff(sorted_angles)
    wrap_gap = sorted_angles[0] + 360.0 - sorted_angles[-1]
    all_gaps = np.append(gaps, wrap_gap)
    gap_index = int(np.argmax(all_gaps))

    start_index = (gap_index + 1) % sorted_angles.size
    end_index = gap_index
    return float(sorted_angles[start_index]), float(sorted_angles[end_index])


def circular_contains(angle: float, start: float, end: float, margin_deg: float) -> bool:
    start = (start + margin_deg) % 360.0
    end = (end - margin_deg) % 360.0
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def mask_by_hsv(
    hsv: np.ndarray,
    hue_min: int,
    hue_max: int,
    sat_min: int,
    val_min: int,
) -> np.ndarray:
    if hue_min <= hue_max:
        return cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max, 255, 255))
    lower = cv2.inRange(hsv, (hue_min, sat_min, val_min), (179, 255, 255))
    upper = cv2.inRange(hsv, (0, sat_min, val_min), (hue_max, 255, 255))
    return cv2.bitwise_or(lower, upper)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def ring_filter(mask: np.ndarray, cx: float, cy: float, radius: float, min_ratio: float, max_ratio: float) -> np.ndarray:
    height, width = mask.shape[:2]
    ys, xs = np.indices((height, width))
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    keep = (distance >= radius * min_ratio) & (distance <= radius * max_ratio)
    return np.where(keep, mask, 0).astype(np.uint8)


def find_prompt_circle(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    height, width = frame_bgr.shape[:2]
    fallback = (width / 2.0, height / 2.0, min(width, height) * 0.42)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.medianBlur(gray, 5)
    min_radius = max(18, int(min(width, height) * 0.12))
    max_radius = max(min_radius + 4, int(min(width, height) * 0.42))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, min(width, height) * 0.35),
        param1=70,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return fallback

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    best_circle: Optional[tuple[float, float, float]] = None
    best_score = -1.0

    for x, y, radius in np.round(circles[0, :]).astype(int):
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius), 255, 3)
        ring_sat = cv2.mean(saturation, mask=mask)[0]
        center_distance = math.hypot(x - width / 2.0, y - height / 2.0)
        score = ring_sat + radius * 0.25 - center_distance * 0.02
        if score > best_score:
            best_score = score
            best_circle = (float(x), float(y), float(radius))

    return best_circle or fallback


def build_digit_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]
    for digit in "0123456789":
        variants: list[np.ndarray] = []
        for font in fonts:
            for scale in (1.7, 2.0, 2.3):
                for thickness in (2, 3, 4):
                    canvas = np.zeros((96, 72), dtype=np.uint8)
                    (tw, th), baseline = cv2.getTextSize(digit, font, scale, thickness)
                    x = max(0, (canvas.shape[1] - tw) // 2)
                    y = max(th + 2, (canvas.shape[0] + th) // 2 - baseline)
                    cv2.putText(canvas, digit, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                    variants.append(normalize_digit_bitmap(canvas))
        templates[digit] = variants
    return templates


def normalize_digit_bitmap(bitmap: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(bitmap, 80, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((64, 48), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y : y + h, x : x + w]
    canvas = np.zeros((72, 56), dtype=np.uint8)
    scale = min(48 / max(w, 1), 64 / max(h, 1))
    resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    yy = (canvas.shape[0] - resized.shape[0]) // 2
    xx = (canvas.shape[1] - resized.shape[1]) // 2
    canvas[yy : yy + resized.shape[0], xx : xx + resized.shape[1]] = resized
    return cv2.resize(canvas, (48, 64), interpolation=cv2.INTER_AREA)


def best_center_component(binary: np.ndarray) -> Optional[np.ndarray]:
    binary = clean_mask(binary)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None

    height, width = binary.shape[:2]
    center = np.array([width / 2.0, height / 2.0])
    best_label: Optional[int] = None
    best_score = -1.0

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 20:
            continue
        if h < height * 0.22 or w < width * 0.05:
            continue
        if h > height * 0.85 or w > width * 0.75:
            continue

        component_center = np.array(centroids[label])
        distance = float(np.linalg.norm(component_center - center))
        if distance > min(width, height) * 0.42:
            continue

        score = float(area) - distance * 3.0
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    component = np.where(labels == best_label, 255, 0).astype(np.uint8)
    component = cv2.dilate(component, np.ones((2, 2), np.uint8), iterations=1)
    return component


def score_digit_bitmap(candidate: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[Optional[str], float]:
    candidate = normalize_digit_bitmap(candidate)
    candidate_bool = candidate > 64

    best_digit: Optional[str] = None
    best_score = -1.0
    for digit, variants in templates.items():
        for template in variants:
            template_bool = template > 64
            intersection = np.logical_and(candidate_bool, template_bool).sum()
            union = np.logical_or(candidate_bool, template_bool).sum()
            if union == 0:
                continue
            foreground_iou = float(intersection / union)
            pixel_agreement = float((candidate_bool == template_bool).mean())
            score = foreground_iou * 0.75 + pixel_agreement * 0.25
            if score > best_score:
                best_score = score
                best_digit = digit

    return best_digit, best_score


def read_digit(
    frame_bgr: np.ndarray,
    templates: dict[str, list[np.ndarray]],
    cx: float,
    cy: float,
    radius: float,
) -> tuple[Optional[str], float]:
    height, width = frame_bgr.shape[:2]
    size = max(36, int(radius * 1.12))
    x1 = max(0, int(cx - size // 2))
    x2 = min(width, int(cx + size // 2))
    y1 = max(0, int(cy - size // 2))
    y2 = min(height, int(cy + size // 2))
    crop = frame_bgr[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv[:, :, 1], 0, 120)
    light_digit = cv2.bitwise_and(cv2.inRange(gray, 145, 255), low_sat)
    dark_digit = cv2.inRange(gray, 0, 105)
    _, otsu_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_digit: Optional[str] = None
    best_score = -1.0
    for mask in (light_digit, dark_digit, otsu_light, otsu_dark):
        component = best_center_component(mask)
        if component is None:
            continue
        digit, score = score_digit_bitmap(component, templates)
        if score > best_score:
            best_digit = digit
            best_score = score

    if best_score < 0.43:
        return None, max(best_score, 0.0)
    return best_digit, best_score


def detect(frame_bgr: np.ndarray, args: argparse.Namespace, templates: dict[str, list[np.ndarray]]) -> Detection:
    cx, cy, radius = find_prompt_circle(frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    red1 = mask_by_hsv(hsv, args.red_hue_min, args.red_hue_max, args.red_sat_min, args.red_val_min)
    blue = mask_by_hsv(hsv, args.blue_hue_min, args.blue_hue_max, args.blue_sat_min, args.blue_val_min)

    red = ring_filter(clean_mask(red1), cx, cy, radius, args.ring_min_ratio, args.ring_max_ratio)
    blue = ring_filter(clean_mask(blue), cx, cy, radius, args.ring_min_ratio, args.ring_max_ratio)

    red_ys, red_xs = np.where(red > 0)
    blue_ys, blue_xs = np.where(blue > 0)
    red_pixels = int(red_xs.size)
    blue_pixels = int(blue_xs.size)

    red_angle = None
    target_start = None
    target_end = None
    inside = False

    if red_pixels >= args.min_red_pixels and blue_pixels >= args.min_blue_pixels:
        red_angles = angle_clockwise_from_top(red_xs, red_ys, cx, cy)
        blue_angles = angle_clockwise_from_top(blue_xs, blue_ys, cx, cy)
        red_angle = circular_mean_deg(red_angles)
        target_start, target_end = circular_arc_span(blue_angles)
        if red_angle is not None and target_start is not None and target_end is not None:
            inside = circular_contains(red_angle, target_start, target_end, args.inside_margin_deg)

    digit, confidence = read_digit(frame_bgr, templates, cx, cy, radius)
    return Detection(
        inside=inside,
        digit=digit,
        red_angle=red_angle,
        target_start=target_start,
        target_end=target_end,
        digit_confidence=confidence,
        red_pixels=red_pixels,
        blue_pixels=blue_pixels,
        center_x=cx,
        center_y=cy,
        radius=radius,
    )


def draw_debug(frame: np.ndarray, detection: Detection, active: bool) -> np.ndarray:
    out = frame.copy()
    cx = int(detection.center_x)
    cy = int(detection.center_y)
    radius = int(detection.radius)

    cv2.circle(out, (cx, cy), radius, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

    if detection.target_start is not None and detection.target_end is not None:
        cv2.ellipse(
            out,
            (cx, cy),
            (radius, radius),
            -90,
            detection.target_start,
            detection.target_end,
            (255, 180, 0),
            2,
        )
    if detection.red_angle is not None:
        radians = math.radians(detection.red_angle)
        x2 = int(cx + math.sin(radians) * radius)
        y2 = int(cy - math.cos(radians) * radius)
        cv2.line(out, (cx, cy), (x2, y2), (0, 0, 255), 2)

    status = "ON" if active else "OFF"
    color = (0, 220, 0) if active else (120, 120, 120)
    text = (
        f"{status} inside={detection.inside} digit={detection.digit or '?'} "
        f"conf={detection.digit_confidence:.2f} red={detection.red_pixels} blue={detection.blue_pixels}"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def select_roi() -> dict[str, int]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    roi = cv2.selectROI("Select circle prompt, then press Enter", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select circle prompt, then press Enter")
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise SystemExit("No region selected.")
    return {"left": x + monitor["left"], "top": y + monitor["top"], "width": w, "height": h}


def parse_roi(raw: Optional[str]) -> Optional[dict[str, int]]:
    if not raw:
        return None
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be left,top,width,height")
    left, top, width, height = parts
    return {"left": left, "top": top, "width": width, "height": height}


def parse_delay_ms(raw: str) -> tuple[float, float]:
    value = raw.strip().lower()
    if value in {"off", "none", "0", "0,0"}:
        return 0.0, 0.0

    normalized = value.replace("-", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) == 1:
        delay = float(parts[0])
        if delay < 0:
            raise argparse.ArgumentTypeError("--delay-ms cannot be negative")
        return delay, delay
    if len(parts) == 2:
        low = float(parts[0])
        high = float(parts[1])
        if low < 0 or high < 0:
            raise argparse.ArgumentTypeError("--delay-ms cannot be negative")
        if low > high:
            low, high = high, low
        return low, high
    raise argparse.ArgumentTypeError("--delay-ms must be off, one number, or min,max")


def load_config(path: str) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    parser = configparser.ConfigParser()
    parser.read(config_path)
    if "settings" not in parser:
        return {}
    return dict(parser["settings"])


def config_bool(config: dict[str, str], key: str, default: bool) -> bool:
    if key not in config:
        return default
    return config[key].strip().lower() in {"1", "yes", "true", "on"}


def bgr_from_hsv(hue: int, sat: int, val: int) -> tuple[int, int, int]:
    hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_arc(
    frame: np.ndarray,
    start_deg: float,
    end_deg: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.38)
    axes = (radius, radius)
    if start_deg <= end_deg:
        cv2.ellipse(frame, center, axes, -90, start_deg, end_deg, color, thickness, cv2.LINE_AA)
    else:
        cv2.ellipse(frame, center, axes, -90, start_deg, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, center, axes, -90, 0, end_deg, color, thickness, cv2.LINE_AA)


def make_test_prompt(
    digit: str,
    red_angle: float,
    red_color: tuple[int, int, int],
    blue_color: tuple[int, int, int],
    center_color: tuple[int, int, int] = (18, 18, 18),
    digit_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    frame = np.zeros((220, 220, 3), dtype=np.uint8)
    cv2.circle(frame, (110, 110), 84, (36, 46, 40), 10, cv2.LINE_AA)
    cv2.circle(frame, (110, 110), 73, center_color, -1, cv2.LINE_AA)
    draw_arc(frame, 125, 205, blue_color, 14)
    draw_arc(frame, red_angle - 7, red_angle + 7, red_color, 14)
    (tw, th), baseline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(
        frame,
        digit,
        ((frame.shape[1] - tw) // 2, (frame.shape[0] + th) // 2 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        digit_color,
        4,
        cv2.LINE_AA,
    )
    return frame


def run_self_test(args: argparse.Namespace, templates: dict[str, list[np.ndarray]]) -> int:
    red_variants = [
        bgr_from_hsv(0, 230, 245),
        bgr_from_hsv(176, 210, 220),
        bgr_from_hsv(4, 180, 190),
    ]
    blue_variants = [
        bgr_from_hsv(96, 220, 245),
        bgr_from_hsv(112, 190, 220),
        bgr_from_hsv(130, 170, 205),
    ]
    failures: list[str] = []
    total = 0

    for digit in "123456789":
        for red_color in red_variants:
            for blue_color in blue_variants:
                styles = [
                    ("light-on-dark", (18, 18, 18), (245, 245, 245)),
                    ("dark-on-yellow", (191, 232, 255), (5, 5, 5)),
                ]
                for style_name, center_color, digit_color in styles:
                    total += 1
                    inside_frame = make_test_prompt(digit, 160, red_color, blue_color, center_color, digit_color)
                    inside_detection = detect(inside_frame, args, templates)
                    if not inside_detection.inside or inside_detection.digit != digit:
                        failures.append(
                            f"{style_name} inside digit={digit}: got inside={inside_detection.inside}, "
                            f"digit={inside_detection.digit}, conf={inside_detection.digit_confidence:.2f}"
                        )

                    total += 1
                    outside_frame = make_test_prompt(digit, 45, red_color, blue_color, center_color, digit_color)
                    outside_detection = detect(outside_frame, args, templates)
                    if outside_detection.inside:
                        failures.append(f"{style_name} outside digit={digit}: detected inside at angle {outside_detection.red_angle}")

    passed = total - len(failures)
    print(f"[circle-key-assist] self-test: {passed}/{total} checks passed")
    for failure in failures[:12]:
        print(f"[circle-key-assist] FAIL {failure}")
    if len(failures) > 12:
        print(f"[circle-key-assist] ...and {len(failures) - 12} more failures")
    return 1 if failures else 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toggleable screen-capture key assist for a local timing prompt.")
    parser.add_argument("--config", default="config.ini", help="Path to config file. Default: config.ini.")
    parser.add_argument("--roi", help="Capture region as left,top,width,height. If omitted, you select it on startup.")
    parser.add_argument("--toggle-key", help="Global hotkey for on/off. Overrides config.ini.")
    parser.add_argument("--quit-key", help="Global hotkey to quit. Overrides config.ini.")
    parser.add_argument("--debug", action="store_true", help="Show a live debug window.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended key presses without pressing keys.")
    parser.add_argument("--self-test", action="store_true", help="Run generated shade/digit detection checks and exit.")
    parser.add_argument("--fps", type=float, help="Capture loop target FPS. Overrides config.ini.")
    parser.add_argument(
        "--delay-ms",
        type=parse_delay_ms,
        help="Optional random delay before pressing, e.g. off, 12, or 8,24. Overrides config.ini.",
    )
    parser.add_argument("--press-cooldown", type=float, help="Minimum seconds between key presses. Overrides config.ini.")
    parser.add_argument("--inside-margin-deg", type=float, help="Positive expands timing, negative waits deeper inside target. Overrides config.ini.")
    parser.add_argument("--ring-min-ratio", type=float, help="Inner ring filter as fraction of detected prompt radius. Overrides config.ini.")
    parser.add_argument("--ring-max-ratio", type=float, help="Outer ring filter as fraction of detected prompt radius. Overrides config.ini.")
    parser.add_argument("--min-red-pixels", type=int)
    parser.add_argument("--min-blue-pixels", type=int)
    parser.add_argument("--red-hue-min", type=int)
    parser.add_argument("--red-hue-max", type=int)
    parser.add_argument("--red-sat-min", type=int)
    parser.add_argument("--red-val-min", type=int)
    parser.add_argument("--blue-hue-min", type=int)
    parser.add_argument("--blue-hue-max", type=int)
    parser.add_argument("--blue-sat-min", type=int)
    parser.add_argument("--blue-val-min", type=int)
    return parser


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    defaults = {
        "toggle_key": "f8",
        "quit_key": "f12",
        "fps": 60.0,
        "delay_ms": (0.0, 0.0),
        "press_cooldown": 0.35,
        "inside_margin_deg": 0.0,
        "ring_min_ratio": 0.55,
        "ring_max_ratio": 1.35,
        "min_red_pixels": 12,
        "min_blue_pixels": 25,
        "red_hue_min": 170,
        "red_hue_max": 10,
        "red_sat_min": 70,
        "red_val_min": 70,
        "blue_hue_min": 90,
        "blue_hue_max": 135,
        "blue_sat_min": 50,
        "blue_val_min": 60,
    }

    for key, default in defaults.items():
        current = getattr(args, key)
        if current is not None:
            continue
        raw = config.get(key, None)
        if raw is None:
            setattr(args, key, default)
        elif key == "delay_ms":
            setattr(args, key, parse_delay_ms(raw))
        elif isinstance(default, int):
            setattr(args, key, int(raw))
        elif isinstance(default, float):
            setattr(args, key, float(raw))
        else:
            setattr(args, key, raw)

    if not args.debug:
        args.debug = config_bool(config, "debug", False)
    if not args.dry_run:
        args.dry_run = config_bool(config, "dry_run", False)

    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = apply_config_defaults(make_parser().parse_args(argv))
    state = RuntimeState()
    templates = build_digit_templates()

    if args.self_test:
        return run_self_test(args, templates)

    key_controller = keyboard.Controller()

    def toggle() -> None:
        state.active = not state.active
        print(f"[circle-key-assist] {'ACTIVE' if state.active else 'inactive'}")

    def quit_app() -> None:
        state.quitting = True
        print("[circle-key-assist] quitting")

    hotkeys = keyboard.GlobalHotKeys(
        {
            hotkey_name(args.toggle_key): toggle,
            hotkey_name(args.quit_key): quit_app,
        }
    )
    hotkey_thread = threading.Thread(target=hotkeys.run, daemon=True)
    hotkey_thread.start()

    roi = parse_roi(args.roi) or select_roi()
    print(f"[circle-key-assist] ROI: {roi['left']},{roi['top']},{roi['width']},{roi['height']}")
    print(f"[circle-key-assist] Toggle: {args.toggle_key.upper()} | Quit: {args.quit_key.upper()}")

    frame_delay = 1.0 / max(args.fps, 1.0)
    was_inside = False
    last_press_at = 0.0
    pending_press: Optional[PendingPress] = None
    delay_low_ms, delay_high_ms = args.delay_ms
    if delay_high_ms > 0:
        print(f"[circle-key-assist] Random press delay: {delay_low_ms:.0f}-{delay_high_ms:.0f} ms")

    with mss.mss() as sct:
        while not state.quitting:
            loop_start = time.perf_counter()
            shot = np.array(sct.grab(roi))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            detection = detect(frame, args, templates)

            entered = detection.inside and not was_inside
            now = time.perf_counter()

            if not state.active or not detection.inside:
                pending_press = None

            if state.active and entered and detection.digit is not None:
                if delay_high_ms > 0:
                    delay_seconds = random.uniform(delay_low_ms, delay_high_ms) / 1000.0
                    pending_press = PendingPress(detection.digit, now + delay_seconds)
                    print(f"[circle-key-assist] queued {detection.digit} in {delay_seconds * 1000:.0f} ms")
                else:
                    pending_press = PendingPress(detection.digit, now)

            if pending_press is not None and now >= pending_press.press_at and now - last_press_at >= args.press_cooldown:
                if args.dry_run:
                    print(f"[circle-key-assist] would press {pending_press.digit}")
                else:
                    key_controller.press(pending_press.digit)
                    key_controller.release(pending_press.digit)
                    print(f"[circle-key-assist] pressed {pending_press.digit}")
                last_press_at = now
                pending_press = None

            was_inside = detection.inside

            if args.debug:
                cv2.imshow("Circle Key Assist Debug", draw_debug(frame, detection, state.active))
                if cv2.waitKey(1) & 0xFF == 27:
                    state.quitting = True
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    hotkeys.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
