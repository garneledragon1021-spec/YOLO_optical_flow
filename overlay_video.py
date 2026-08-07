#!/usr/bin/env python3
"""座標 CSV を元動画へ重ねて描画し、動きの速さを色で表示する。

CSV の x/y 列を自動検出し、必要に応じて BB 中心をクリックで指定する。
座標差分から BB 周りの角速度を求め、軌跡・現在位置・速度情報を動画へ描く。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from video_io import make_video_writer


Point = tuple[float, float] | None
NORMAL_COLOR = (0, 255, 0)
FAST_COLOR = (0, 0, 255)
UNKNOWN_COLOR = (255, 255, 0)


def filtered_suffixes(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    return coordinate_suffixes(fieldnames)


def coordinate_suffixes(fieldnames: list[str]) -> list[str]:
    """x_SUFFIX と y_SUFFIX がそろっている座標列の suffix を返す。"""
    return [
        field.removeprefix("x_")
        for field in fieldnames
        if field.startswith("x_") and f"y_{field.removeprefix('x_')}" in fieldnames
    ]


def select_gui_inputs(args: argparse.Namespace) -> argparse.Namespace:
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(f"tkinter is unavailable; specify --video and --csv ({exc})")

    root = tk.Tk()
    root.withdraw()
    try:
        if args.video is None:
            selected = filedialog.askopenfilename(
                title="元動画を選択",
                filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
            )
            if not selected:
                raise SystemExit("動画選択がキャンセルされました")
            args.video = Path(selected)

        if args.csv is None:
            selected = filedialog.askopenfilename(
                title="フィルター後座標CSVを選択",
                initialdir="results",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not selected:
                raise SystemExit("CSV選択がキャンセルされました")
            args.csv = Path(selected)

        if args.bb_x is None and args.bb_y is None and args.bb_frame is None:
            args.bb_frame = simpledialog.askinteger(
                "BB中心を指定",
                "BB中心をクリック選択するフレーム番号を入力してください:",
                initialvalue=0,
                minvalue=0,
                parent=root,
            )
            if args.bb_frame is None:
                raise SystemExit("BB中心の指定がキャンセルされました")

        suffixes = filtered_suffixes(args.csv)
        if args.suffix is None and "savgol" not in suffixes and len(suffixes) > 1:
            selected = simpledialog.askstring(
                "表示するフィルターを選択",
                "表示する列のsuffixを入力してください:\n" + ", ".join(suffixes),
                initialvalue="w11" if "w11" in suffixes else suffixes[0],
                parent=root,
            )
            if not selected:
                raise SystemExit("フィルター選択がキャンセルされました")
            args.suffix = selected

        if args.output is None:
            suffix = args.suffix.removeprefix("_") if args.suffix else "filtered"
            selected = filedialog.asksaveasfilename(
                title="出力動画の保存先を選択",
                initialdir=str(args.video.parent),
                initialfile=f"{args.video.stem}_{suffix}.mp4",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
            )
            if not selected:
                raise SystemExit("出力先選択がキャンセルされました")
            args.output = Path(selected)
    finally:
        root.destroy()
    return args


def select_bb_center_from_frame(video_path: Path, frame_index: int) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video for BB selection: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read --bb-frame {frame_index} from video")

    height, width = frame.shape[:2]
    preview_scale = min(1.0, 1280 / width, 800 / height)
    if preview_scale < 1.0:
        preview = cv2.resize(
            frame,
            (round(width * preview_scale), round(height * preview_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = frame

    selected: list[tuple[float, float]] = []
    window_name = f"Click BB center - frame {frame_index} (Enter: confirm, Esc: cancel)"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(x / preview_scale, y / preview_scale)]

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            display = preview.copy()
            if selected:
                center = (
                    round(selected[0][0] * preview_scale),
                    round(selected[0][1] * preview_scale),
                )
                cv2.drawMarker(
                    display,
                    center,
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=24,
                    thickness=3,
                    line_type=cv2.LINE_AA,
                )
            cv2.putText(
                display,
                "Click BB center, then press Enter",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and selected:
                return selected[0]
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("BB center selection was cancelled")
    finally:
        cv2.destroyWindow(window_name)


def parse_point(row: dict[str, str], x_column: str, y_column: str) -> Point:
    x_value = row.get(x_column, "")
    y_value = row.get(y_column, "")
    if not x_value or not y_value:
        return None
    x = float(x_value)
    y = float(y_value)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def select_filtered_columns(fieldnames: list[str], suffix: str | None) -> tuple[str, str, str]:
    if suffix:
        normalized = suffix.removeprefix("_")
        if normalized == "raw":
            x_column, y_column, label = "x", "y", "raw"
        else:
            x_column = f"x_{normalized}"
            y_column = f"y_{normalized}"
            label = normalized
    elif "x_savgol" in fieldnames and "y_savgol" in fieldnames:
        x_column, y_column, label = "x_savgol", "y_savgol", "savgol"
    else:
        candidates = coordinate_suffixes(fieldnames)
        if len(candidates) > 1:
            available = ", ".join(candidates) or "none"
            raise SystemExit(
                "multiple filtered coordinate columns found; "
                f"use --suffix with one of: {available}"
            )
        if len(candidates) == 1:
            label = candidates[0]
            x_column, y_column = f"x_{label}", f"y_{label}"
        elif "x" in fieldnames and "y" in fieldnames:
            x_column, y_column, label = "x", "y", "raw"
        else:
            raise SystemExit("CSV must contain x/y or x_<suffix>/y_<suffix> coordinate columns")

    if x_column not in fieldnames or y_column not in fieldnames:
        raise SystemExit(f"CSV does not contain {x_column}/{y_column}")
    return x_column, y_column, label


def load_coordinates(
    csv_path: Path,
    suffix: str | None,
) -> tuple[dict[int, Point], dict[int, Point], str, str, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "frame" not in fieldnames:
            raise SystemExit("CSV must contain a frame column")
        filtered_x, filtered_y, label = select_filtered_columns(fieldnames, suffix)
        has_raw = "x" in fieldnames and "y" in fieldnames
        filtered: dict[int, Point] = {}
        raw: dict[int, Point] = {}
        for row in reader:
            frame_index = int(float(row["frame"]))
            filtered[frame_index] = parse_point(row, filtered_x, filtered_y)
            if has_raw:
                raw[frame_index] = parse_point(row, "x", "y")
    return filtered, raw, label, filtered_x, filtered_y


def calculate_angular_speeds(
    points: dict[int, Point],
    center: tuple[float, float],
    fps: float,
) -> dict[int, float]:
    """Return absolute angular velocity around center in degrees per second."""
    angular_speeds: dict[int, float] = {}
    run_frames: list[int] = []
    run_points: list[tuple[float, float]] = []

    def finish_run() -> None:
        if len(run_frames) < 3:
            return
        frames = np.asarray(run_frames, dtype=np.float64)
        coordinates = np.asarray(run_points, dtype=np.float64)
        bb_centered_x = coordinates[:, 0] - center[0]
        bb_centered_y = center[1] - coordinates[:, 1]
        angle = np.unwrap(np.mod(np.arctan2(bb_centered_y, bb_centered_x), 2.0 * np.pi))
        omega_deg_per_second = np.abs(np.degrees(np.gradient(angle, frames))) * fps
        angular_speeds.update(zip(run_frames, omega_deg_per_second.tolist()))

    previous_frame: int | None = None
    for frame_index, point in sorted(points.items()):
        if point is None or (previous_frame is not None and frame_index != previous_frame + 1):
            finish_run()
            run_frames.clear()
            run_points.clear()
        if point is not None:
            run_frames.append(frame_index)
            run_points.append(point)
            previous_frame = frame_index
        else:
            previous_frame = None
    finish_run()
    return angular_speeds


def angular_speed_color(angular_speed: float | None, threshold: float) -> tuple[int, int, int]:
    if angular_speed is None or not math.isfinite(angular_speed):
        return UNKNOWN_COLOR
    if threshold == 0:
        return FAST_COLOR
    ratio = min(max(angular_speed / threshold, 0.0), 1.0)
    return (
        0,
        round(NORMAL_COLOR[1] * (1.0 - ratio)),
        round(FAST_COLOR[2] * ratio),
    )


def draw_point(frame, point: Point, color: tuple[int, int, int], marker_size: int, thickness: int) -> None:
    if point is None:
        return
    cv2.drawMarker(
        frame,
        (round(point[0]), round(point[1])),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )


def draw_rotation_radius(
    frame,
    center: tuple[float, float],
    point: Point,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    center_pixel = round(center[0]), round(center[1])
    cv2.drawMarker(
        frame,
        center_pixel,
        (255, 255, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if point is not None:
        cv2.line(
            frame,
            center_pixel,
            (round(point[0]), round(point[1])),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_trail(
    frame,
    trail: deque[tuple[Point, tuple[int, int, int]]],
    thickness: int,
) -> None:
    previous = None
    for point, color in trail:
        if point is None:
            previous = None
            continue
        current = round(point[0]), round(point[1])
        if previous is not None:
            cv2.line(frame, previous, current, color, thickness, cv2.LINE_AA)
        previous = current


def render_video(args: argparse.Namespace) -> None:
    filtered, raw, label, filtered_x, filtered_y = load_coordinates(args.csv, args.suffix)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = make_video_writer(cap, args.output)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    bb_center = args.bb_x, args.bb_y
    angular_speeds = calculate_angular_speeds(filtered, bb_center, fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trail: deque[tuple[Point, tuple[int, int, int]]] = deque(maxlen=args.trail_length)
    frame_index = 0

    print(f"video: {args.video}")
    print(f"csv: {args.csv}")
    print(f"coordinate columns: {filtered_x}, {filtered_y}")
    print(f"BB center: ({args.bb_x:g}, {args.bb_y:g})")
    print(f"fully red angular speed: >= {args.angular_speed_threshold:g} deg/s")
    print(f"output: {args.output}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            filtered_point = filtered.get(frame_index)
            angular_speed = angular_speeds.get(frame_index)
            filtered_color = angular_speed_color(angular_speed, args.angular_speed_threshold)
            trail.append((filtered_point, filtered_color))
            draw_rotation_radius(
                frame,
                bb_center,
                filtered_point,
                filtered_color,
                args.trail_thickness,
            )
            if args.trail_length > 1:
                draw_trail(frame, trail, args.trail_thickness)
            if args.show_raw:
                draw_point(frame, raw.get(frame_index), (0, 0, 255), args.marker_size, args.thickness)
            draw_point(frame, filtered_point, filtered_color, args.marker_size, args.thickness)

            cv2.putText(
                frame,
                f"filtered={label} frame={frame_index} "
                f"|omega|={angular_speed:.1f} deg/s" if angular_speed is not None
                else f"filtered={label} frame={frame_index} |omega|=n/a",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if args.show_raw:
                cv2.putText(
                    frame,
                    "green=slow red=fast/raw cyan=unknown",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            if frame_index % args.progress_every == 0:
                print(f"frame {frame_index}/{total_frames}")
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    print(f"saved: {args.output}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay filtered CSV coordinates on a video")
    parser.add_argument("--video", type=Path, help="source video path; GUI selection opens when omitted")
    parser.add_argument("--csv", type=Path, help="coordinate CSV path; GUI selection opens when omitted")
    parser.add_argument(
        "--suffix",
        help="filtered column suffix, for example savgol or w11; auto-detected when omitted",
    )
    parser.add_argument("-o", "--output", type=Path, help="output MP4 path")
    parser.add_argument("--show-raw", action="store_true", help="also draw raw x/y coordinates in red")
    parser.add_argument("--bb-x", type=float, help="BB center x coordinate in pixels")
    parser.add_argument("--bb-y", type=float, help="BB center y coordinate in pixels")
    parser.add_argument(
        "--bb-frame",
        type=int,
        help="show this video frame and select the BB center by clicking",
    )
    parser.add_argument("--trail-length", type=int, default=30, help="filtered trajectory length in frames")
    parser.add_argument("--marker-size", type=int, default=20, help="cross marker size")
    parser.add_argument("--thickness", type=int, default=3, help="cross marker thickness")
    parser.add_argument("--trail-thickness", type=int, default=2, help="trajectory line thickness")
    parser.add_argument(
        "--angular-speed-threshold",
        type=float,
        default=180.0,
        help="absolute angular speed in deg/s at which the point/trail becomes fully red",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="print progress every N frames")
    args = parser.parse_args(argv)

    if args.video is None or args.csv is None:
        args = select_gui_inputs(args)
    elif args.output is None:
        suffix = args.suffix.removeprefix("_") if args.suffix else "filtered"
        args.output = args.video.with_name(f"{args.video.stem}_{suffix}.mp4")
    has_bb_coordinates = args.bb_x is not None or args.bb_y is not None
    if has_bb_coordinates and (args.bb_x is None or args.bb_y is None):
        parser.error("--bb-x and --bb-y must be specified together")
    if has_bb_coordinates and args.bb_frame is not None:
        parser.error("specify either --bb-x/--bb-y or --bb-frame, not both")
    if not has_bb_coordinates and args.bb_frame is None:
        parser.error("specify --bb-x/--bb-y or --bb-frame")
    if has_bb_coordinates and (not math.isfinite(args.bb_x) or not math.isfinite(args.bb_y)):
        parser.error("--bb-x and --bb-y must be finite")
    if args.bb_frame is not None:
        if args.bb_frame < 0:
            parser.error("--bb-frame must be >= 0")
        args.bb_x, args.bb_y = select_bb_center_from_frame(args.video, args.bb_frame)
    for name in ("trail_length", "marker_size", "thickness", "trail_thickness", "progress_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.angular_speed_threshold < 0:
        parser.error("--angular-speed-threshold must be >= 0")
    return args


def main(argv: list[str]) -> int:
    render_video(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
