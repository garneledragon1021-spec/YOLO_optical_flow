#!/usr/bin/env python3
"""RTMPose WholeBodyとLucas-Kanade法で動画のつま先を追跡する。"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from rtmlib import Wholebody
    from scipy.signal import savgol_filter
except ModuleNotFoundError as exc:
    print("依存関係が不足しています。次のコマンドで環境を整えてから再実行してください。", file=sys.stderr)
    print("  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

from video_io import make_video_writer

RESULTS_DIR = Path("results")

# COCO-WholeBodyのキーポイント番号。
# body: 0-16、足: 17-22（左3点、右3点）
TOE_LANDMARKS = {
    "left": 17,   # left_big_toe
    "right": 20,  # right_big_toe
}

POSE_SEGMENT_LANDMARKS = {
    "left": (13, 15, 17),   # left knee, left ankle, left big toe
    "right": (14, 16, 20),  # right knee, right ankle, right big toe
}


def infer_keypoints(
    model: Wholebody,
    frame: np.ndarray,
    pose_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RTMPoseを実行し、元画像座標系のキーポイントとスコアを返す。"""
    if pose_scale != 1.0:
        pose_frame = cv2.resize(
            frame,
            None,
            fx=pose_scale,
            fy=pose_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        pose_frame = frame

    keypoints, scores = model(pose_frame)
    xy = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if pose_scale != 1.0 and xy.size:
        xy /= pose_scale
    return xy, confidence


def detect_toe(
    model: Wholebody,
    frame: np.ndarray,
    landmark_index: int,
    confidence: float,
    pose_scale: float,
) -> tuple[float, float] | None:
    """最も信頼度の高い人物の指定つま先を返す。"""
    keypoints, scores = infer_keypoints(model, frame, pose_scale)
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return None
    if keypoints.shape[1] <= landmark_index:
        return None
    if scores.shape != keypoints.shape[:2]:
        scores = np.ones(keypoints.shape[:2], dtype=np.float32)

    toe_scores = scores[:, landmark_index]
    valid = toe_scores >= confidence
    if not np.any(valid):
        return None
    person_index = int(np.argmax(np.where(valid, toe_scores, -1.0)))
    x, y = keypoints[person_index, landmark_index]
    return float(x), float(y)


def detect_pose_leg_points(
    model: Wholebody,
    frame: np.ndarray,
    side: str,
    pose_scale: float,
    confidence: float,
) -> dict[str, tuple[float, float]] | None:
    """膝・足首・つま先を取得し、骨格整合性チェック用に返す。"""
    keypoints, scores = infer_keypoints(model, frame, pose_scale)
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return None

    indices = POSE_SEGMENT_LANDMARKS[side]
    if keypoints.shape[1] <= max(indices) or scores.shape != keypoints.shape[:2]:
        return None

    # 指定側のつま先スコアが最も高い人物を対象にする。
    toe_index = indices[-1]
    candidate_scores = scores[:, toe_index]
    person_index = int(np.argmax(candidate_scores))
    if np.any(scores[person_index, list(indices)] < confidence):
        return None

    names = ("knee", "ankle", "toe")
    return {
        name: (
            float(keypoints[person_index, index, 0]),
            float(keypoints[person_index, index, 1]),
        )
        for name, index in zip(names, indices)
    }


def leg_geometry_is_valid(
    points: dict[str, tuple[float, float]],
    previous: tuple[float, float, float] | None,
    tolerance: float = 0.30,
) -> tuple[bool, tuple[float, float, float]]:
    """膝-足首、足首-つま先の長さの急変を検出する。"""
    knee = np.asarray(points["knee"], dtype=np.float64)
    ankle = np.asarray(points["ankle"], dtype=np.float64)
    toe = np.asarray(points["toe"], dtype=np.float64)
    lengths = (
        float(np.linalg.norm(knee - ankle)),
        float(np.linalg.norm(ankle - toe)),
        float(np.linalg.norm(knee - toe)),
    )
    if min(lengths) < 8.0:
        return False, lengths
    if previous is None:
        return True, lengths
    ratios = [current / old for current, old in zip(lengths, previous) if old > 1e-6]
    return all((1.0 - tolerance) <= ratio <= (1.0 + tolerance) for ratio in ratios), lengths


def crop_bounds(
    point: tuple[float, float],
    width: int,
    height: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    x, y = point
    half = crop_size // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(width, int(x) + half)
    y2 = min(height, int(y) + half)
    return x1, y1, x2, y2


def track_toe_single(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    point: tuple[float, float],
    crop_size: int,
) -> tuple[float, float] | None:
    """つま先位置をLucas-Kanade法で1点追跡する。"""
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
    crop_size: int,
    delegate: str,
    source: str,
) -> None:
    """採用点、Pose点、光学フロー点を注釈動画へ描画する。"""
    for candidate, color, marker in (
        (pose_point, (255, 0, 0), "P"),
        (flow_point, (0, 0, 255), "F"),
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
    """--video省略時にファイル選択ダイアログを表示する。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI環境依存
        raise SystemExit(
            "tkinterが使えません。--videoで動画ファイルを指定してください。"
            f" ({exc})"
        )

    root = tk.Tk()
    root.withdraw()
    video_files = filedialog.askopenfilenames(
        title="RTMPose flowで処理する動画を選択",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
    )
    root.destroy()
    return list(video_files)


def output_paths(
    video_path: str,
    out_dir: Path,
    write_video: bool,
) -> tuple[Path, Path | None]:
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
    """有効な座標列へSavitzky-Golayフィルタを適用する。"""
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

    for index, point in zip(valid_indices, valid_points):
        filtered[index] = float(point[0]), float(point[1])
    return filtered


def write_results_csv(
    csv_path: Path,
    rows: list[tuple[
        int,
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
            "x_savgol",
            "y_savgol",
            "source",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
        ])
        for (frame_idx, point, pose_point, flow_point, source, bounds), filtered_point in zip(rows, filtered_points):
            raw_values = ["", ""] if point is None else [f"{point[0]:.6f}", f"{point[1]:.6f}"]
            pose_values = ["", ""] if pose_point is None else [f"{pose_point[0]:.6f}", f"{pose_point[1]:.6f}"]
            flow_values = ["", ""] if flow_point is None else [f"{flow_point[0]:.6f}", f"{flow_point[1]:.6f}"]
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
                *filtered_values,
                source,
                *crop_values,
            ])


def create_model(args: argparse.Namespace) -> tuple[Wholebody, str]:
    model = Wholebody(
        mode=args.rtmpose_mode,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )
    delegate = f"RTMPose WholeBody ({args.rtmpose_mode}, {args.backend}, {args.device})"
    return model, delegate


def process_video(
    args: argparse.Namespace,
    video_path: str,
    model: Wholebody,
    delegate: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open: {video_path}")
        return

    csv_path, annotated_path = output_paths(video_path, args.out_dir, args.write_video)
    video_writer = make_video_writer(cap, annotated_path) if annotated_path is not None else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    previous_gray = None
    point = None
    source = "none"
    frame_idx = 0
    rows = []
    previous_leg_geometry = None

    print(f"Processing: {video_path}")
    print(f"  csv: {csv_path}")
    print(f"  RTMPose: mode={args.rtmpose_mode}, backend={args.backend}, device={args.device}")
    print(f"  detection interval: every {args.detect_every} frames")
    print(f"  Savitzky-Golay: window={args.savgol_window}, polyorder={args.savgol_polyorder}")
    if annotated_path is not None:
        print(f"  annotated video: {annotated_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            needs_pose = point is None or frame_idx % args.detect_every == 0
            pose_point = None
            flow_point = None

            if not needs_pose and previous_gray is not None:
                tracked = track_toe_single(previous_gray, gray, point, args.crop_size)  # type: ignore
                if tracked is not None:
                    flow_point = tracked
                    point = tracked
                    source = "flow"
                else:
                    needs_pose = True

            if needs_pose:
                leg_points = detect_pose_leg_points(
                    model,
                    frame,
                    args.side,
                    args.pose_scale,
                    args.confidence,
                )
                if leg_points is not None:
                    geometry_valid, current_geometry = leg_geometry_is_valid(
                        leg_points,
                        previous_leg_geometry,
                    )
                else:
                    geometry_valid = False
                    current_geometry = None

                if leg_points is not None and geometry_valid:
                    previous_leg_geometry = current_geometry
                    pose_point = leg_points["toe"]
                    point = pose_point
                    source = "pose"
                else:
                    # 一時的な誤検出で直前の追跡点を消さない。
                    source = "pose_geometry_rejected" if leg_points is not None else "none"
                    if point is None:
                        source = "none"

            bounds = None
            if point is not None:
                bounds = crop_bounds(point, frame.shape[1], frame.shape[0], args.crop_size)

            rows.append((frame_idx, point, pose_point, flow_point, source, bounds))

            if video_writer is not None:
                draw_overlay(
                    frame,
                    point,
                    pose_point,
                    flow_point,
                    args.crop_size,
                    delegate,
                    source,
                )
                video_writer.write(frame)

            if frame_idx % args.progress_every == 0:
                print(f"  frame {frame_idx}/{total_frames} source={source}")

            previous_gray = gray
            frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    write_results_csv(csv_path, rows, args.savgol_window, args.savgol_polyorder)
    print(f"  done: {frame_idx} frames")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process videos with RTMPose WholeBody toe flow")
    parser.add_argument("--check", action="store_true", help="initialize RTMPose and exit")
    parser.add_argument(
        "--video",
        action="append",
        help="video file path to process. Repeat for multiple files. If omitted, a file picker opens.",
    )
    parser.add_argument("--side", choices=sorted(TOE_LANDMARKS), default="left", help="toe landmark to track")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--device", default="cpu", help="inference device: cpu, cuda, or mps")
    parser.add_argument(
        "--backend",
        choices=["onnxruntime", "opencv", "openvino"],
        default="onnxruntime",
        help="RTMPose inference backend",
    )
    parser.add_argument(
        "--rtmpose-mode",
        choices=["performance", "balanced", "lightweight"],
        default="lightweight",
        help="WholeBody model preset; models are downloaded on first run",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="keypoint confidence threshold")
    parser.add_argument("--detect-every", type=int, default=30, help="run RTMPose every N frames")
    parser.add_argument("--crop-size", type=int, default=160, help="optical-flow crop size around the toe")
    parser.add_argument("--pose-scale", type=float, default=1.0, help="scale input before RTMPose refresh")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--write-video", action="store_true", help="write an annotated MP4 alongside the CSV")
    parser.add_argument("--progress-every", type=int, default=30, help="print progress every N frames")
    parser.add_argument("--savgol-window", type=int, default=11, help="Savitzky-Golay window length")
    parser.add_argument("--savgol-polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    args = parser.parse_args(argv)

    if args.cpu:
        args.device = "cpu"
    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if not (0.1 <= args.pose_scale <= 1.0):
        parser.error("--pose-scale must be between 0.1 and 1.0")
    if not (0.0 < args.confidence <= 1.0):
        parser.error("--confidence must be between 0 and 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.savgol_window < 3 or args.savgol_window % 2 == 0:
        parser.error("--savgol-window must be an odd integer >= 3")
    if args.savgol_polyorder < 0 or args.savgol_polyorder >= args.savgol_window - 1:
        parser.error(
            "--savgol-polyorder must be >= 0 and at least 2 less than --savgol-window"
        )
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, delegate = create_model(args)
    if args.check:
        print(f"delegate: {delegate}")
        print("RTMPose WholeBody initialization: OK")
        return 0

    videos = args.video if args.video else select_videos()
    if not videos:
        print("No files selected. 終了します。")
        return 0

    print(f"delegate: {delegate}")
    for video_path in videos:
        process_video(args, video_path, model, delegate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
