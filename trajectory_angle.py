#!/usr/bin/env python3
"""座標 CSV から移動方向（軌跡の heading）の角速度を計算する。

画像座標の y 軸は下向きなので、計算時だけ y を反転して数学座標系に直す。
その後、移動方向の unwrap と Savitzky-Golay 微分を行い、ノイズを抑えた
角速度を CSV に書き出す。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def coordinate_pairs(fieldnames: list[str]) -> list[tuple[str, str, str]]:
    """CSV ヘッダーから対応する x/y 列を探し、(ラベル, x列, y列) を返す。"""
    fields = set(fieldnames)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    if "x" in fields and "y" in fields and ("x", "y") not in seen:
        pairs.append(("raw", "x", "y"))
        seen.add(("x", "y"))

    for x_column in fieldnames:
        match = re.fullmatch(r"x(.*)", x_column)
        if not match:
            continue
        suffix = match.group(1)
        y_column = f"y{suffix}"
        if y_column in fields and (x_column, y_column) not in seen:
            pairs.append((suffix.removeprefix("_") or "raw", x_column, y_column))
            seen.add((x_column, y_column))
    return pairs


def resolve_input_path(input_path: Path) -> Path:
    """Resolve an input CSV path from the current working directory or the project results folder."""
    if input_path.is_absolute():
        return input_path

    repo_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / input_path,
        repo_root / input_path,
        repo_root / "results" / input_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    for pattern in (input_path.name, f"*{input_path.stem}*.csv", "*.csv"):
        matches = sorted(
            [p.resolve() for p in repo_root.rglob(pattern) if p.is_file() and p.suffix.lower() == ".csv"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    results_dir = repo_root / "results"
    if results_dir.exists():
        matches = sorted(
            [p.resolve() for p in results_dir.glob("*.csv") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(f"input CSV not found: {input_path}")


def resolve_output_path(output_path: Path) -> Path:
    """Resolve an output CSV path under the project results folder by default."""
    if output_path.is_absolute():
        return output_path

    repo_root = Path(__file__).resolve().parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if output_path.parts and output_path.parts[0] == "results":
        return (repo_root / output_path).resolve()
    return (results_dir / output_path).resolve()


def trajectory_angular_velocity(
    frames: np.ndarray,
    points: np.ndarray,
    min_speed: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """heading、未平滑/平滑角速度、速度をフレーム単位で返す。

    速度が極端に小さいフレームでは移動方向が不定になるため、結果を NaN
    にして CSV 上では空欄として扱えるようにする。
    """
    frame_steps = np.diff(frames)
    if not np.allclose(frame_steps, frame_steps[0]):
        raise ValueError("Savitzky-Golay angular velocity requires equally spaced frames")
    frame_step = float(frame_steps[0])
    dx = np.gradient(points[:, 0], frames)
    # Image y increases downward. Negate dy so positive angles are counterclockwise.
    dy = np.gradient(-points[:, 1], frames)
    speed = np.hypot(dx, dy)
    heading = np.unwrap(np.arctan2(dy, dx))
    angular_velocity = np.gradient(heading, frames)
    angular_velocity_savgol = savgol_filter(
        heading,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
        deriv=1,
        delta=frame_step,
    )
    invalid = speed < min_speed
    heading[invalid] = np.nan
    angular_velocity[invalid] = np.nan
    angular_velocity_savgol[invalid] = np.nan
    return heading, angular_velocity, angular_velocity_savgol, speed


def format_value(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9f}"


def process_csv(
    input_path: Path,
    output_path: Path,
    fps: float | None,
    min_speed: float,
    savgol_window: int,
    polyorder: int,
) -> None:
    try:
        resolved_input_path = resolve_input_path(input_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    resolved_output_path = resolve_output_path(output_path)
    print(f"using input CSV: {resolved_input_path}")
    print(f"writing output CSV: {resolved_output_path}")
    with resolved_input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "frame" not in fieldnames:
        raise SystemExit("input CSV must contain a frame column")

    if any(
        name.startswith("heading_deg_")
        or name.startswith("omega_")
        or name.startswith("speed_px_per_frame_")
        for name in fieldnames
    ):
        raise SystemExit(
            "input CSV appears to be an angular-velocity output file. "
            "Please pass the original coordinate CSV produced by toe_mp.py or toe_flow.py."
        )

    pairs = coordinate_pairs(fieldnames)
    if not pairs:
        raise SystemExit(
            "input CSV does not contain x/y coordinate columns. "
            "Please pass the original coordinate CSV produced by toe_mp.py or toe_flow.py, "
            "not an angular-velocity output file."
        )

    frames = np.asarray([float(row["frame"]) for row in rows], dtype=np.float64)
    if savgol_window > len(frames):
        raise SystemExit(f"--omega-savgol-window must be <= the number of rows ({len(frames)})")

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, x_column, y_column in pairs:
        valid_rows = [row for row in rows if row.get(x_column) not in (None, "") and row.get(y_column) not in (None, "")]
        if not valid_rows:
            print(f"skip {label}: no valid coordinates", file=sys.stderr)
            continue
        if len(valid_rows) < 3:
            print(f"skip {label}: too few valid coordinates", file=sys.stderr)
            continue

        valid_frames = np.asarray([float(row["frame"]) for row in valid_rows], dtype=np.float64)
        points = np.asarray(
            [[float(row[x_column]), float(row[y_column])] for row in valid_rows],
            dtype=np.float64,
        )
        if len(valid_frames) > 1:
            frame_step = valid_frames[1] - valid_frames[0]
            if not np.allclose(np.diff(valid_frames), frame_step):
                valid_frames = np.arange(len(valid_frames), dtype=np.float64)
        results[label] = trajectory_angular_velocity(
            valid_frames,
            points,
            min_speed,
            savgol_window,
            polyorder,
        )

    headers = ["frame"]
    for label in results:
        headers.extend([
            f"heading_deg_{label}",
            f"omega_deg_per_frame_{label}",
            f"omega_rad_per_frame_{label}",
            f"omega_savgol_deg_per_frame_{label}",
            f"omega_savgol_rad_per_frame_{label}",
            f"speed_px_per_frame_{label}",
        ])
        if fps is not None:
            headers.extend([
                f"omega_deg_per_s_{label}",
                f"omega_rad_per_s_{label}",
                f"omega_savgol_deg_per_s_{label}",
                f"omega_savgol_rad_per_s_{label}",
            ])

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for index, frame in enumerate(valid_frames):
            output = [format_value(frame)]
            for heading, omega, omega_savgol, speed in results.values():
                output.extend([
                    format_value(np.degrees(heading[index])),
                    format_value(np.degrees(omega[index])),
                    format_value(omega[index]),
                    format_value(np.degrees(omega_savgol[index])),
                    format_value(omega_savgol[index]),
                    format_value(speed[index]),
                ])
                if fps is not None:
                    output.extend([
                        format_value(np.degrees(omega[index]) * fps),
                        format_value(omega[index] * fps),
                        format_value(np.degrees(omega_savgol[index]) * fps),
                        format_value(omega_savgol[index] * fps),
                    ])
            writer.writerow(output)

    print(f"saved: {resolved_output_path}")
    print(f"coordinate sets: {', '.join(results)}")
    print("angular velocity: trajectory heading change, positive=counterclockwise")
    print(f"angular velocity Savitzky-Golay: window={savgol_window}, polyorder={polyorder}")
    if fps is None:
        print("FPS was not specified; output units are per frame.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate trajectory-heading angular velocity from CSV")
    parser.add_argument("input", type=Path, help="CSV containing frame and x/y coordinate pairs")
    parser.add_argument("-o", "--output", type=Path, help="output CSV path")
    parser.add_argument("--fps", type=float, help="video FPS; enables per-second angular velocity columns")
    parser.add_argument(
        "--omega-savgol-window",
        type=int,
        default=11,
        help="Savitzky-Golay window for angular velocity differentiation",
    )
    parser.add_argument(
        "--omega-savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order for angular velocity differentiation",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=1e-6,
        help="set heading/angular velocity blank below this speed in px/frame",
    )
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.min_speed < 0:
        parser.error("--min-speed must be >= 0")
    if args.omega_savgol_window < 3 or args.omega_savgol_window % 2 == 0:
        parser.error("--omega-savgol-window must be an odd integer >= 3")
    if (
        args.omega_savgol_polyorder < 1
        or args.omega_savgol_polyorder >= args.omega_savgol_window
    ):
        parser.error(
            "--omega-savgol-polyorder must be >= 1 and less than --omega-savgol-window"
        )
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_angular_velocity.csv")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    process_csv(
        args.input,
        args.output,
        args.fps,
        args.min_speed,
        args.omega_savgol_window,
        args.omega_savgol_polyorder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
