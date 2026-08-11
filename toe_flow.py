#!/usr/bin/env python3
#toe_flow.py
"""YOLO Poseとオプティカルフローで動画を処理する。

YOLO Pose は指定間隔で足首位置を更新する時だけ使う。
更新の間は、つま先周辺の小さいクロップ内で Lucas-Kanade optical flow
により点を追跡する。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from ultralytics import YOLO
from video_io import make_video_writer

RESULTS_DIR = Path("results")

TOE_LANDMARKS = {
    "left": 15,   # left ankle (COCO pose index)
    "right": 16,  # right ankle (COCO pose index)
}

def detect_toe(
    model: YOLO,
    frame: np.ndarray,
    landmark_index: int,
    confidence: float,
    imgsz: int,
    device: str | int | None = None,
) -> tuple[float, float] | None:
    """YOLO Poseから左右の足首キーポイントを取得する。"""
    results = model.predict(source=frame, conf=confidence, imgsz=imgsz, device=device, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None
    keypoints = results[0].keypoints
    xy = keypoints.xy.cpu().numpy()
    conf = keypoints.conf
    conf_np = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
    valid = conf_np[:, landmark_index] >= confidence
    if not np.any(valid):
        return None
    person_index = int(np.argmax(np.where(valid, conf_np[:, landmark_index], -1.0)))
    x, y = xy[person_index, landmark_index]
    return float(x), float(y)


def crop_bounds(point: tuple[float, float], width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    # optical flow の追跡範囲を、直前のつま先位置を中心に切り出す。
    x, y = point
    half = crop_size // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(width, int(x) + half)
    y2 = min(height, int(y) + half)
    return x1, y1, x2, y2


def track_toe(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    point: tuple[float, float],
    crop_size: int,
) -> tuple[float, float] | None:
    """つま先周辺の複数特徴点を追跡し、中央値の移動量を返す。"""
    # 1点だけでは背景やペダルを誤追跡しやすいため、つま先周辺の
    # 複数特徴点を追跡し、外れ値に比較的強い中央値で移動量を求める。
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = crop_bounds(point, width, height, crop_size)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    prev_crop = prev_gray[y1:y2, x1:x2]
    crop = gray[y1:y2, x1:x2]
    prev_points = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=25,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if prev_points is None or len(prev_points) < 3:
        return None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        crop,
        prev_points,
        None, # type: ignore
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    ) # type: ignore
    if status is None or next_points is None:
        return None

    valid = status.reshape(-1) == 1
    if int(np.count_nonzero(valid)) < 3:
        return None

    old_valid = prev_points.reshape(-1, 2)[valid]
    new_valid = next_points.reshape(-1, 2)[valid]
    displacements = new_valid - old_valid

    # 極端に大きい移動量を除外してから中央値を取る。
    displacement_limit = max(25.0, crop_size * 0.35)
    displacement_lengths = np.linalg.norm(displacements, axis=1)
    stable = displacement_lengths <= displacement_limit
    if int(np.count_nonzero(stable)) < 3:
        return None

    median_displacement = np.median(displacements[stable], axis=0)
    next_x = float(point[0] + median_displacement[0])
    next_y = float(point[1] + median_displacement[1])
    if not (0 <= next_x < width and 0 <= next_y < height):
        return None
    return next_x, next_y


def track_toe_single(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    point: tuple[float, float],
    crop_size: int,
) -> tuple[float, float] | None:
    """つま先位置そのものを1点だけLucas-Kanadeで追跡する。"""
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = crop_bounds(point, width, height, crop_size)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    prev_crop = prev_gray[y1:y2, x1:x2]
    crop = gray[y1:y2, x1:x2]
    prev_point = np.array([[[point[0] - x1, point[1] - y1]]], dtype=np.float32)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        crop,
        prev_point,
        None,  # type: ignore
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )  # type: ignore
    if status is None or next_points is None or status[0][0] != 1:
        return None

    next_x = float(next_points[0][0][0] + x1)
    next_y = float(next_points[0][0][1] + y1)
    if not (0 <= next_x < width and 0 <= next_y < height):
        return None
    return next_x, next_y


def draw_overlay(
    frame: np.ndarray,
    point: tuple[float, float] | None,
    pose_point: tuple[float, float] | None,
    flow_point: tuple[float, float] | None,
    periodic_point: tuple[float, float] | None,
    crop_size: int,
    delegate: str,
    source: str,
) -> None:
    # 注釈付き動画用に、採用点と各推定元を色分けして描画する。
    # 緑=最終採用点、青=Pose、赤=光学フロー、紫=周期予測。
    for candidate, color, marker in (
        (pose_point, (255, 0, 0), "P"),
        (flow_point, (0, 0, 255), "F"),
        (periodic_point, (255, 0, 255), "T"),
    ):
        if candidate is not None:
            cv2.drawMarker(
                frame,
                (int(candidate[0]), int(candidate[1])),
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=14,
                thickness=2,
            )
            cv2.putText(
                frame,
                marker,
                (int(candidate[0]) + 8, int(candidate[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if point is not None:
        x, y = point
        x1, y1, x2, y2 = crop_bounds(point, frame.shape[1], frame.shape[0], crop_size)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 255), 2)
        cv2.drawMarker(
            frame,
            (int(x), int(y)),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
    cv2.putText(
        frame,
        f"delegate={delegate} source={source}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def select_videos() -> list[str]:
    # --video が省略された場合はファイル選択ダイアログを使う。
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(
            "tkinter が使えません。--video で動画ファイルを指定してください。"
            f" ({exc})"
        )

    root = tk.Tk()
    root.withdraw()
    video_files = filedialog.askopenfilenames(
        title="OpenPose flow で処理する動画を選択",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
    )
    root.destroy()
    return list(video_files)


def output_paths(video_path: str, out_dir: Path, write_video: bool) -> tuple[Path, Path | None]:
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = out_dir / f"{base}_toe_flow.csv"
    if not write_video:
        return csv_path, None
    return csv_path, out_dir / f"{base}_toe_flow.mp4"


def apply_savgol(
    points: list[tuple[float, float] | None],
    window_length: int,
    polyorder: int,
) -> list[tuple[float, float] | None]:
    """欠損を除いた有効座標列に Savitzky-Golay フィルタを適用する。"""
    filtered: list[tuple[float, float] | None] = [None] * len(points)
    valid_indices = [index for index, point in enumerate(points) if point is not None]
    if not valid_indices:
        return filtered

    valid_points = np.asarray([points[index] for index in valid_indices], dtype=np.float64)
    effective_window = min(window_length, len(valid_points))
    if effective_window % 2 == 0:
        effective_window -= 1

    if effective_window > polyorder:
        valid_points = savgol_filter(
            valid_points,
            window_length=effective_window,
            polyorder=polyorder,
            axis=0,
        )

    for index, point in zip(valid_indices, valid_points): # type: ignore
        filtered[index] = float(point[0]), float(point[1])

    return filtered


def write_results_csv(
    csv_path: Path,
    rows: list[tuple[
        int,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
        str,
        tuple[int, int, int, int] | None,
    ]],
    window_length: int,
    polyorder: int,
) -> None:
    filtered_points = apply_savgol([row[1] for row in rows], window_length, polyorder)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "x",
            "y",
            "pose_x",
            "pose_y",
            "flow_x",
            "flow_y",
            "periodic_x",
            "periodic_y",
            "x_savgol",
            "y_savgol",
            "source",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
        ])
        for (frame_idx, point, pose_point, flow_point, periodic_point, source, bounds), filtered_point in zip(rows, filtered_points): # type: ignore
            raw_values = ["", ""] if point is None else [f"{point[0]:.6f}", f"{point[1]:.6f}"]
            pose_values = ["", ""] if pose_point is None else [f"{pose_point[0]:.6f}", f"{pose_point[1]:.6f}"]
            flow_values = ["", ""] if flow_point is None else [f"{flow_point[0]:.6f}", f"{flow_point[1]:.6f}"]
            periodic_values = ["", ""] if periodic_point is None else [f"{periodic_point[0]:.6f}", f"{periodic_point[1]:.6f}"]
            filtered_values = (
                ["", ""]
                if filtered_point is None
                else [f"{filtered_point[0]:.6f}", f"{filtered_point[1]:.6f}"]
            )
            crop_values = ["", "", "", ""] if bounds is None else list(bounds)
            writer.writerow([
                frame_idx,
                *raw_values,
                *pose_values,
                *flow_values,
                *periodic_values,
                *filtered_values,
                source,
                *crop_values,
            ])


def process_video(args: argparse.Namespace, video_path: str, model: YOLO, delegate: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open: {video_path}")
        return

    csv_path, annotated_path = output_paths(video_path, args.out_dir, args.write_video)
    video_writer = make_video_writer(cap, annotated_path) if annotated_path is not None else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmark_index = TOE_LANDMARKS[args.side]
    prev_gray = None
    point = None
    source = "none"
    frame_idx = 0
    rows = []

    print(f"Processing: {video_path}")
    print(f"  csv: {csv_path}")
    print(f"  Savitzky-Golay: window={args.savgol_window}, polyorder={args.savgol_polyorder}")
    if annotated_path is not None:
        print(f"  annotated video: {annotated_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 追跡点がない場合、または指定間隔に達した場合はYOLOで再検出する。
        # それ以外の中間フレームは optical flow で追跡する。
        needs_pose = point is None or frame_idx % args.detect_every == 0

        pose_point = None
        flow_point = None
        periodic_point = None

        if not needs_pose and prev_gray is not None:
            tracked = track_toe_single(prev_gray, gray, point, args.crop_size) # type: ignore
            if tracked is not None:
                flow_point = tracked
                point = tracked
                source = "flow"
            else:
                # optical flow が点を見失った場合は、同じフレームで Pose に戻して復帰する。
                needs_pose = True

        if needs_pose:
            detected = detect_toe(
                model,
                frame,
                landmark_index,
                args.confidence,
                args.imgsz,
                args.device,
            )
            if detected is not None:
                pose_point = detected
                point = detected
                source = "pose"
            else:
                source = "none"
                if point is None:
                    source = "none"

        bounds = None
        if point is not None:
            bounds = crop_bounds(point, frame.shape[1], frame.shape[0], args.crop_size)
        rows.append((frame_idx, point, pose_point, flow_point, periodic_point, source, bounds))

        if video_writer is not None:
            draw_overlay(
                frame,
                point,
                pose_point,
                flow_point,
                periodic_point,
                args.crop_size,
                delegate,
                source,
            )
            video_writer.write(frame)

        if frame_idx % args.progress_every == 0:
            print(f"  frame {frame_idx}/{total_frames} source={source}")

        prev_gray = gray
        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    write_results_csv(csv_path, rows, args.savgol_window, args.savgol_polyorder)
    print(f"  done: {frame_idx} frames")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process videos with YOLO Pose and optical flow")
    parser.add_argument(
        "--video",
        action="append",
        help="video file path to process. Repeat for multiple files. If omitted, a file picker opens.",
    )
    parser.add_argument("--side", choices=sorted(TOE_LANDMARKS), default="left", help="toe landmark to track")
    parser.add_argument("--cpu", action="store_true", help="use CPU inference")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO Pose model name or path")
    parser.add_argument("--confidence", type=float, default=0.35, help="keypoint confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--detect-every", type=int, default=15, help="run YOLO Pose every N frames")
    parser.add_argument("--crop-size", type=int, default=160, help="optical-flow crop size around the toe")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--write-video", action="store_true", help="write an annotated MP4 alongside the CSV")
    parser.add_argument("--progress-every", type=int, default=30, help="print progress every N frames")
    parser.add_argument("--savgol-window", type=int, default=11, help="Savitzky-Golay window length")
    parser.add_argument("--savgol-polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    args = parser.parse_args(argv)

    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if not (0.0 < args.confidence <= 1.0):
        parser.error("--confidence must be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be >= 32")
    if args.cpu:
        args.device = "cpu"
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.savgol_window < 3 or args.savgol_window % 2 == 0:
        parser.error("--savgol-window must be an odd integer >= 3")
    if args.savgol_polyorder < 0 or args.savgol_polyorder >= args.savgol_window - 1:
        parser.error(
            "--savgol-polyorder must be >= 0 and at least 2 less than --savgol-window; "
            "polyorder = window - 1 reproduces the input without smoothing"
        )
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = args.video if args.video else select_videos()
    if not videos:
        print("No files selected. 終了します。")
        return 0

    model = YOLO(args.model)
    delegate = f"YOLO ({args.device})"
    print(f"delegate: {delegate}")
    print(f"model: {args.model}")
    for video_path in videos:
        process_video(args, video_path, model, delegate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
