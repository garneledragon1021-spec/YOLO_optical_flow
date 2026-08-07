#!/usr/bin/env python3
"""BB とペダルのマーカー座標から速度・角速度を計算する。

欠損していない連続区間ごとに Savitzky-Golay 平滑化と数値微分を行うため、
動画の途中で検出に失敗したフレームがあっても、欠損をまたいだ不自然な速度を
計算しない。座標は画像座標系（右が +x、下が +y）として扱う。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from video_io import make_video_writer


SLOW_COLOR = (0, 255, 0)
FAST_COLOR = (0, 0, 255)
UNKNOWN_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)


def parse_point(row: dict[str, str], x_column: str, y_column: str) -> tuple[float, float] | None:
    """CSV の x/y 列を有限な浮動小数点座標へ変換する。"""
    x_value = row.get(x_column, "")
    y_value = row.get(y_column, "")
    if not x_value or not y_value:
        return None
    x = float(x_value)
    y = float(y_value)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def format_value(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9f}"


def short_value(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def valid_runs(valid: np.ndarray) -> list[slice]:
    """有効フレームが連続する区間を slice のリストとして返す。"""
    runs = []
    start = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append(slice(start, index))
            start = None
    if start is not None:
        runs.append(slice(start, len(valid)))
    return runs


def smooth_points(
    frames: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """有効区間ごとに座標を Savitzky-Golay フィルターで平滑化する。"""
    smoothed = points.copy()
    for run in valid_runs(valid):
        run_frames = frames[run]
        run_points = points[run]
        if len(run_points) < 3:
            continue
        frame_steps = np.diff(run_frames)
        if len(frame_steps) and not np.allclose(frame_steps, frame_steps[0]):
            continue
        effective_window = min(window_length, len(run_points))
        if effective_window % 2 == 0:
            effective_window -= 1
        if effective_window <= polyorder:
            continue
        smoothed[run] = savgol_filter(
            run_points,
            window_length=effective_window,
            polyorder=polyorder,
            axis=0,
        )
    return smoothed


def gradient_by_run(values: np.ndarray, frames: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """欠損をまたがず、各連続区間内だけ数値微分する。"""
    gradient = np.full_like(values, np.nan, dtype=np.float64)
    for run in valid_runs(valid):
        if run.stop - run.start < 2:
            continue
        gradient[run] = np.gradient(values[run], frames[run], axis=0)
    return gradient


def bb_centered_points(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """BB 座標を原点に移したペダル位置を返す。"""
    return points - centers


def calculate_velocities(
    frames: np.ndarray,
    bb_points: np.ndarray,
    pedal_points: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """BB速度、ペダル速度、相対速度、半径方向/接線方向の量をまとめて計算する。"""
    bb_velocity = gradient_by_run(bb_points, frames, valid)
    pedal_velocity = gradient_by_run(pedal_points, frames, valid)
    bb_centered = bb_centered_points(pedal_points, bb_points)
    relative_velocity = pedal_velocity - bb_velocity

    radius = np.full(len(frames), np.nan, dtype=np.float64)
    angle = np.full(len(frames), np.nan, dtype=np.float64)
    omega = np.full(len(frames), np.nan, dtype=np.float64)
    radial_speed = np.full(len(frames), np.nan, dtype=np.float64)

    radius[valid] = np.hypot(bb_centered[valid, 0], bb_centered[valid, 1])
    crank_angle = np.full(len(frames), np.nan, dtype=np.float64)
    unwrapped_angle = np.full(len(frames), np.nan, dtype=np.float64)
    # Crank angle from the BB-centered pedal coordinate: atan2(y, x).
    # MediaPipe/OpenCV image coordinates are kept as-is: +x right, +y downward.
    crank_angle[valid] = np.mod(
        np.arctan2(bb_centered[valid, 1], bb_centered[valid, 0]),
        2.0 * np.pi,
    )

    for run in valid_runs(valid):
        if run.stop - run.start < 2:
            continue
        unwrapped_angle[run] = np.unwrap(crank_angle[run])
        omega[run] = np.gradient(unwrapped_angle[run], frames[run])
        radial_speed[run] = np.gradient(radius[run], frames[run])
    angle[valid] = crank_angle[valid]

    tangential_speed = radius * omega

    return {
        "bb_centered_x": bb_centered[:, 0],
        "bb_centered_y": bb_centered[:, 1],
        "bb_vx": bb_velocity[:, 0],
        "bb_vy": bb_velocity[:, 1],
        "bb_speed": np.hypot(bb_velocity[:, 0], bb_velocity[:, 1]),
        "pedal_vx": pedal_velocity[:, 0],
        "pedal_vy": pedal_velocity[:, 1],
        "pedal_speed": np.hypot(pedal_velocity[:, 0], pedal_velocity[:, 1]),
        "relative_vx": relative_velocity[:, 0],
        "relative_vy": relative_velocity[:, 1],
        "relative_speed": np.hypot(relative_velocity[:, 0], relative_velocity[:, 1]),
        "radius": radius,
        "angle": angle,
        "unwrapped_angle": unwrapped_angle,
        "omega": omega,
        "radial_speed": radial_speed,
        "tangential_speed": tangential_speed,
    }


def draw_text_with_shadow(
    image,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def angular_speed_values(velocities: dict[str, np.ndarray], fps: float | None) -> tuple[np.ndarray, str]:
    angular_speed = np.abs(np.degrees(velocities["omega"]))
    if fps is None:
        return angular_speed, "deg/frame"
    return angular_speed * fps, "deg/s"


def valid_index_runs(valid: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append(np.arange(start, index))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(valid)))
    return runs


def auto_angular_speed_threshold(
    angles: np.ndarray,
    angular_speeds: np.ndarray,
) -> tuple[float, str]:
    finite = np.isfinite(angles) & np.isfinite(angular_speeds)
    if not np.any(finite):
        return 1.0, "fallback"

    cycle_peaks: list[float] = []
    for run_indices in valid_index_runs(finite):
        if len(run_indices) < 3:
            continue
        run_angles = angles[run_indices]
        run_speeds = angular_speeds[run_indices]
        direction = np.nanmedian(np.diff(run_angles))
        if not math.isfinite(direction) or math.isclose(direction, 0.0):
            continue
        progress = (run_angles - run_angles[0]) * (1.0 if direction > 0 else -1.0)
        cycle_numbers = np.floor(progress / (2.0 * math.pi)).astype(int)
        for cycle_number in sorted(set(cycle_numbers.tolist())):
            cycle_mask = cycle_numbers == cycle_number
            cycle_progress = progress[cycle_mask]
            if len(cycle_progress) < 3:
                continue
            if np.nanmax(cycle_progress) - np.nanmin(cycle_progress) < 1.5 * math.pi:
                continue
            cycle_speeds = run_speeds[cycle_mask]
            cycle_speeds = cycle_speeds[np.isfinite(cycle_speeds)]
            if len(cycle_speeds) == 0:
                continue
            cycle_peaks.append(float(np.nanpercentile(cycle_speeds, 95)))

    if cycle_peaks:
        cycle_threshold = float(np.nanpercentile(cycle_peaks, 90))
        finite_speeds = angular_speeds[np.isfinite(angular_speeds)]
        robust_threshold = float(np.nanpercentile(finite_speeds, 90))
        threshold = min(cycle_threshold, robust_threshold)
        source = f"auto robust p90, cycle peaks n={len(cycle_peaks)}"
    else:
        finite_speeds = angular_speeds[np.isfinite(angular_speeds)]
        threshold = float(np.nanpercentile(finite_speeds, 90))
        source = "auto all-frame p90"

    if not math.isfinite(threshold) or threshold <= 0:
        threshold = 1.0
        source = "fallback"
    return threshold, source


def angular_speed_color(angular_speed: float, threshold: float) -> tuple[int, int, int]:
    if not math.isfinite(angular_speed):
        return UNKNOWN_COLOR
    if threshold <= 0:
        return FAST_COLOR
    ratio = min(max(angular_speed / threshold, 0.0), 1.0)
    return (
        0,
        round(SLOW_COLOR[1] * (1.0 - ratio)),
        round(FAST_COLOR[2] * ratio),
    )


def draw_pedal_marker(
    frame,
    point: tuple[float, float],
    color: tuple[int, int, int],
    marker_size: int,
    marker_thickness: int,
) -> None:
    import cv2

    center = round(point[0]), round(point[1])
    radius = max(marker_size // 2, 3)
    cv2.circle(frame, center, radius + marker_thickness, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, TEXT_COLOR, marker_thickness, cv2.LINE_AA)


def draw_trail(
    frame,
    trail: deque[tuple[tuple[float, float], tuple[int, int, int]]],
    thickness: int,
) -> None:
    import cv2

    previous = None
    for point, color in trail:
        current = round(point[0]), round(point[1])
        if previous is not None:
            cv2.line(frame, previous, current, color, thickness, cv2.LINE_AA)
        previous = current


def render_velocity_marker_video(
    args: argparse.Namespace,
    frames: np.ndarray,
    pedal_points: np.ndarray,
    velocities: dict[str, np.ndarray],
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = make_video_writer(cap, args.output_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    angular_speeds, angular_speed_unit = angular_speed_values(velocities, args.fps)
    if args.angular_speed_threshold is None:
        angular_speed_threshold, threshold_source = auto_angular_speed_threshold(
            velocities["unwrapped_angle"],
            angular_speeds,
        )
    else:
        angular_speed_threshold = args.angular_speed_threshold
        threshold_source = "manual"
    frame_to_row = {round(frame): index for index, frame in enumerate(frames)}
    trail: deque[tuple[tuple[float, float], tuple[int, int, int]]] = deque(maxlen=args.trail_length)
    frame_index = 0
    print(f"marker video: {args.video}")
    print(f"marker output: {args.output_video}")
    print(
        f"fully red angular speed: >= {angular_speed_threshold:g} {angular_speed_unit} "
        f"({threshold_source})"
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row_index = frame_to_row.get(frame_index)
            if row_index is not None:
                point = pedal_points[row_index]
                angular_speed = float(angular_speeds[row_index])
                if np.all(np.isfinite(point)):
                    color = angular_speed_color(angular_speed, angular_speed_threshold)
                    point_tuple = float(point[0]), float(point[1])
                    trail.append((point_tuple, color))
                    if args.trail_length > 1:
                        draw_trail(frame, trail, args.marker_thickness)
                    draw_pedal_marker(
                        frame,
                        point_tuple,
                        color,
                        args.marker_size,
                        args.marker_thickness,
                    )
                label = f"f={frame_index} |omega|={short_value(angular_speed)} {angular_speed_unit}"
            else:
                label = f"f={frame_index} |omega|=n/a"
            draw_text_with_shadow(frame, label, (12, 30), 0.65, TEXT_COLOR, 2)
            writer.write(frame)
            if frame_index % args.progress_every == 0:
                print(f"marker frame {frame_index}/{total_frames}")
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    print(f"saved marker video: {args.output_video}")


def output_headers(fps: float | None) -> list[str]:
    headers = [
        "frame",
        "bb_x",
        "bb_y",
        "pedal_x",
        "pedal_y",
        "bb_centered_x",
        "bb_centered_y",
        "radius_px",
        "angle_deg",
        "bb_vx_px_per_frame",
        "bb_vy_px_per_frame",
        "bb_speed_px_per_frame",
        "pedal_vx_px_per_frame",
        "pedal_vy_px_per_frame",
        "pedal_speed_px_per_frame",
        "relative_vx_px_per_frame",
        "relative_vy_px_per_frame",
        "relative_speed_px_per_frame",
        "radial_speed_px_per_frame",
        "tangential_speed_px_per_frame",
        "omega_deg_per_frame",
        "omega_rad_per_frame",
    ]
    if fps is not None:
        headers.extend([
            "bb_speed_px_per_s",
            "pedal_speed_px_per_s",
            "relative_speed_px_per_s",
            "radial_speed_px_per_s",
            "tangential_speed_px_per_s",
            "omega_deg_per_s",
            "omega_rad_per_s",
            "rpm",
        ])
    return headers


def process_csv(args: argparse.Namespace) -> None:
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = [
        "frame",
        args.bb_x_column,
        args.bb_y_column,
        args.pedal_x_column,
        args.pedal_y_column,
    ]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SystemExit(f"input CSV is missing columns: {', '.join(missing)}")

    frames = np.asarray([float(row["frame"]) for row in rows], dtype=np.float64)
    bb_points = np.full((len(rows), 2), np.nan, dtype=np.float64)
    pedal_points = np.full((len(rows), 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        bb_point = parse_point(row, args.bb_x_column, args.bb_y_column)
        pedal_point = parse_point(row, args.pedal_x_column, args.pedal_y_column)
        if bb_point is None or pedal_point is None:
            continue
        bb_points[index] = bb_point
        pedal_points[index] = pedal_point
        valid[index] = True

    if not np.any(valid):
        raise SystemExit("no rows contain both BB and pedal coordinates")

    if args.savgol_window is not None:
        bb_points = smooth_points(frames, bb_points, valid, args.savgol_window, args.savgol_polyorder)
        pedal_points = smooth_points(frames, pedal_points, valid, args.savgol_window, args.savgol_polyorder)

    velocities = calculate_velocities(frames, bb_points, pedal_points, valid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_headers(args.fps))
        for index, frame in enumerate(frames):
            row = [
                format_value(frame),
                format_value(bb_points[index, 0]),
                format_value(bb_points[index, 1]),
                format_value(pedal_points[index, 0]),
                format_value(pedal_points[index, 1]),
                format_value(velocities["bb_centered_x"][index]),
                format_value(velocities["bb_centered_y"][index]),
                format_value(velocities["radius"][index]),
                format_value(np.degrees(velocities["angle"][index])),
                format_value(velocities["bb_vx"][index]),
                format_value(velocities["bb_vy"][index]),
                format_value(velocities["bb_speed"][index]),
                format_value(velocities["pedal_vx"][index]),
                format_value(velocities["pedal_vy"][index]),
                format_value(velocities["pedal_speed"][index]),
                format_value(velocities["relative_vx"][index]),
                format_value(velocities["relative_vy"][index]),
                format_value(velocities["relative_speed"][index]),
                format_value(velocities["radial_speed"][index]),
                format_value(velocities["tangential_speed"][index]),
                format_value(np.degrees(velocities["omega"][index])),
                format_value(velocities["omega"][index]),
            ]
            if args.fps is not None:
                row.extend([
                    format_value(velocities["bb_speed"][index] * args.fps),
                    format_value(velocities["pedal_speed"][index] * args.fps),
                    format_value(velocities["relative_speed"][index] * args.fps),
                    format_value(velocities["radial_speed"][index] * args.fps),
                    format_value(velocities["tangential_speed"][index] * args.fps),
                    format_value(np.degrees(velocities["omega"][index]) * args.fps),
                    format_value(velocities["omega"][index] * args.fps),
                    format_value(np.degrees(velocities["omega"][index]) * args.fps / 6.0),
                ])
            writer.writerow(row)

    valid_count = int(np.count_nonzero(valid))
    print(f"saved: {args.output}")
    print(f"valid rows: {valid_count}/{len(rows)}")
    print("angle/omega: atan2(pedal_y - bb_y, pedal_x - bb_x), normalized to 0..360 deg")
    if args.fps is None:
        print("FPS was not specified; output units are per frame.")
    if args.video is not None:
        render_velocity_marker_video(args, frames, pedal_points, velocities)


def select_gui_inputs(args: argparse.Namespace, select_video: bool) -> argparse.Namespace:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(f"tkinter is unavailable; specify input CSV ({exc})")

    root = tk.Tk()
    root.withdraw()
    try:
        if args.input is None:
            selected = filedialog.askopenfilename(
                title="ペダル速度を計算するCSVを選択",
                initialdir="results",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not selected:
                raise SystemExit("CSV選択がキャンセルされました")
            args.input = Path(selected)

        if select_video and args.video is None:
            selected = filedialog.askopenfilename(
                title="角速度色付きマーカを重ねる元動画を選択（キャンセルでCSVのみ）",
                filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
            )
            if selected:
                args.video = Path(selected)

        if args.video is not None and args.output_video is None:
            selected = filedialog.asksaveasfilename(
                title="角速度色付きマーカ動画の保存先を選択",
                initialdir=str(args.video.parent),
                initialfile=f"{args.video.stem}_pedal_velocity_marker.mp4",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
            )
            if not selected:
                raise SystemExit("出力動画の保存先選択がキャンセルされました")
            args.output_video = Path(selected)
    finally:
        root.destroy()
    return args


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate pedal velocity from BB and pedal coordinate CSV")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="CSV containing frame, BB, and pedal coordinates; GUI selection opens when omitted",
    )
    parser.add_argument("-o", "--output", type=Path, help="output CSV path")
    parser.add_argument("--fps", type=float, help="video FPS; enables per-second velocity columns and rpm")
    parser.add_argument("--video", type=Path, help="source video path; also writes an angular-velocity marker MP4")
    parser.add_argument("--output-video", type=Path, help="angular-velocity marker MP4 path")
    parser.add_argument("--bb-prefix", default="bb", help="prefix for BB columns, for example bb -> bb_x/bb_y")
    parser.add_argument(
        "--pedal-prefix",
        default="shoe",
        help="prefix for pedal columns. marker.py default uses shoe -> shoe_x/shoe_y",
    )
    parser.add_argument("--bb-x-column", help="explicit BB x column")
    parser.add_argument("--bb-y-column", help="explicit BB y column")
    parser.add_argument("--pedal-x-column", help="explicit pedal x column")
    parser.add_argument("--pedal-y-column", help="explicit pedal y column")
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=11,
        help="odd window length for coordinate smoothing; use 0 to disable",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for coordinate smoothing",
    )
    parser.add_argument(
        "--angular-speed-threshold",
        type=float,
        help="absolute angular speed that becomes fully red; omitted means auto-scale from per-cycle peaks",
    )
    parser.add_argument("--marker-size", type=int, default=18, help="pedal marker point diameter in pixels")
    parser.add_argument("--marker-thickness", type=int, default=2, help="pedal marker outline/trail thickness")
    parser.add_argument("--trail-length", type=int, default=30, help="colored pedal marker trail length in frames")
    parser.add_argument("--progress-every", type=int, default=100, help="print progress every N video frames")
    args = parser.parse_args(argv)

    input_from_gui = args.input is None
    if input_from_gui:
        args = select_gui_inputs(args, select_video=True)
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_pedal_velocity.csv")
    if args.output_video is None and args.video is not None:
        args.output_video = args.video.with_name(f"{args.video.stem}_pedal_velocity_marker.mp4")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.output_video is not None and args.video is None:
        parser.error("--output-video requires --video")

    args.bb_x_column = args.bb_x_column or f"{args.bb_prefix}_x"
    args.bb_y_column = args.bb_y_column or f"{args.bb_prefix}_y"
    args.pedal_x_column = args.pedal_x_column or f"{args.pedal_prefix}_x"
    args.pedal_y_column = args.pedal_y_column or f"{args.pedal_prefix}_y"

    if args.savgol_window == 0:
        args.savgol_window = None
    elif args.savgol_window < 3 or args.savgol_window % 2 == 0:
        parser.error("--savgol-window must be 0 or an odd integer >= 3")
    if args.savgol_polyorder < 0:
        parser.error("--savgol-polyorder must be >= 0")
    if args.savgol_window is not None and args.savgol_polyorder >= args.savgol_window:
        parser.error("--savgol-polyorder must be less than --savgol-window")
    if args.angular_speed_threshold is not None and args.angular_speed_threshold < 0:
        parser.error("--angular-speed-threshold must be >= 0")
    for name in ("marker_size", "marker_thickness", "trail_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    return args


def main(argv: list[str]) -> int:
    process_csv(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
