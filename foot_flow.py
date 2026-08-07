#!/usr/bin/env python3
"""足部の疎な光学フローから、BB 中心まわりの角速度を推定する。

MediaPipe の足部ランドマークで ROI を決め、ROI 内の特徴点を Lucas-Kanade
法で追跡する。前後方向の追跡誤差で外れ値を除き、BB 中心から見た点の回転量を
集約することで、足全体の回転運動を推定する。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from mediapipe_helper import create_pose_landmarker
from overlay_video import angular_speed_color, select_bb_center_from_frame
from video_io import make_video_writer, video_timestamp_ms


FOOT_LANDMARKS = {
    "left": (27, 29, 31),   # ankle, heel, foot index
    "right": (28, 30, 32),
}

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}


def select_video() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(f"tkinter is unavailable; specify --video ({exc})")

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="足部を追跡する動画を選択",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    if not selected:
        raise SystemExit("動画選択がキャンセルされました")
    return Path(selected)


def detect_foot_landmarks(
    landmarker: vision.PoseLandmarker, # type: ignore
    frame: np.ndarray,
    timestamp_ms: int,
    landmark_indices: tuple[int, int, int],
    pose_scale: float,
    person_mask_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """足首・踵・足先の3点と、必要なら人物セグメンテーション mask を返す。

    Pose 推定を縮小画像で行った場合も、ランドマークは元画像のピクセル座標へ
    戻す。visibility が低いフレームは追跡 ROI を誤るため検出なしとする。
    """
    height, width = frame.shape[:2]
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

    frame_rgba = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGBA)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=np.ascontiguousarray(frame_rgba),
    )
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.pose_landmarks:
        return None, None

    points = []
    for index in landmark_indices:
        landmark = result.pose_landmarks[0][index]
        if landmark.visibility is not None and landmark.visibility < 0.35:
            return None, None
        points.append((landmark.x * width, landmark.y * height))

    person_mask = None
    if result.segmentation_masks:
        probabilities = np.squeeze(result.segmentation_masks[0].numpy_view())
        if probabilities.shape != (height, width):
            probabilities = cv2.resize(
                probabilities,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        person_mask = np.where(probabilities >= person_mask_threshold, 255, 0).astype(np.uint8)
        # Remove isolated segmentation noise before intersecting with the foot ROI.
        person_mask = cv2.morphologyEx(
            person_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
    return np.asarray(points, dtype=np.float32), person_mask


def make_foot_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    padding: int,
) -> np.ndarray:
    """3つの足部ランドマークを結んだ、特徴点検出用の厚み付き ROI を作る。"""
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(landmarks).astype(np.int32)
    ankle, heel, toe = points
    line_thickness = max(2 * padding, 1)
    cv2.line(mask, tuple(ankle), tuple(heel), 255, line_thickness, cv2.LINE_AA)
    cv2.line(mask, tuple(heel), tuple(toe), 255, line_thickness, cv2.LINE_AA)
    cv2.line(mask, tuple(toe), tuple(ankle), 255, line_thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(mask, tuple(point), padding, 255, -1, cv2.LINE_AA)
    return mask


def detect_features(
    gray: np.ndarray,
    mask: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
) -> np.ndarray | None:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )


def track_features(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    previous_points: np.ndarray,
    max_forward_backward_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """特徴点を順方向・逆方向へ追跡し、前後整合性のある点だけ返す。"""
    next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_points,
        None, # type: ignore
        **LK_PARAMS,
    ) # type: ignore
    if next_points is None or forward_status is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        next_points,
        None, # type: ignore
        **LK_PARAMS,
    ) # type: ignore
    if backward_points is None or backward_status is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    previous_flat = previous_points.reshape(-1, 2)
    next_flat = next_points.reshape(-1, 2)
    backward_flat = backward_points.reshape(-1, 2)
    fb_error = np.linalg.norm(previous_flat - backward_flat, axis=1)
    valid = (
        (forward_status.reshape(-1) == 1)
        & (backward_status.reshape(-1) == 1)
        & np.isfinite(next_flat).all(axis=1)
        & (fb_error <= max_forward_backward_error)
    )
    if forward_error is not None:
        valid &= np.isfinite(forward_error.reshape(-1))
    return previous_flat[valid], next_flat[valid], fb_error[valid]


def point_angular_velocities(
    previous_points: np.ndarray,
    points: np.ndarray,
    center: tuple[float, float],
    min_flow: float,
    min_radius: float,
    min_abs_angular_velocity: float,
    max_abs_angular_velocity: float,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)

    center_array = np.asarray(center, dtype=np.float64)
    previous_vectors = previous_points.astype(np.float64) - center_array
    vectors = points.astype(np.float64) - center_array
    # BB-centered coordinates: +x right, +y upward.
    previous_vectors[:, 1] *= -1
    vectors[:, 1] *= -1
    radii = np.minimum(
        np.linalg.norm(previous_vectors, axis=1),
        np.linalg.norm(vectors, axis=1),
    )
    flow = np.linalg.norm(points.astype(np.float64) - previous_points.astype(np.float64), axis=1)
    cross = previous_vectors[:, 0] * vectors[:, 1] - previous_vectors[:, 1] * vectors[:, 0]
    dot = np.sum(previous_vectors * vectors, axis=1)
    angular_velocity = np.degrees(np.arctan2(cross, dot))
    valid = (
        (flow >= min_flow)
        & (radii >= min_radius)
        & np.isfinite(angular_velocity)
        & (np.abs(angular_velocity) >= min_abs_angular_velocity)
        & (np.abs(angular_velocity) <= max_abs_angular_velocity)
    )
    return angular_velocity[valid]


def robust_angular_velocity(
    angular_velocities: np.ndarray,
    mad_scale: float,
) -> tuple[float | None, int]:
    if len(angular_velocities) == 0:
        return None, 0
    median = float(np.median(angular_velocities))
    deviations = np.abs(angular_velocities - median)
    mad = float(np.median(deviations))
    if mad == 0:
        inliers = deviations <= 1e-6
    else:
        inliers = deviations <= mad_scale * 1.4826 * mad
    if not np.any(inliers):
        return None, 0
    return float(np.median(angular_velocities[inliers])), int(np.count_nonzero(inliers))


def draw_overlay(
    frame: np.ndarray,
    bb_center: tuple[float, float],
    foot_landmarks: np.ndarray | None,
    feature_mask: np.ndarray | None,
    tracked_points: np.ndarray | None,
    angular_velocity: float | None,
    inlier_count: int,
    fps: float,
    red_at_deg_per_frame: float,
) -> None:
    color = angular_speed_color(
        None if angular_velocity is None else abs(angular_velocity),
        red_at_deg_per_frame,
    )
    center_pixel = round(bb_center[0]), round(bb_center[1])
    cv2.drawMarker(
        frame,
        center_pixel,
        (255, 255, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=20,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if foot_landmarks is not None:
        polygon = np.round(foot_landmarks).astype(np.int32)
        cv2.polylines(frame, [polygon], True, (255, 180, 0), 2, cv2.LINE_AA)
    if feature_mask is not None:
        contours, _ = cv2.findContours(feature_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 1, cv2.LINE_AA)
    if tracked_points is not None:
        for point in tracked_points.reshape(-1, 2):
            cv2.circle(frame, tuple(np.round(point).astype(int)), 3, color, -1, cv2.LINE_AA)

    if angular_velocity is None:
        text = f"omega=n/a points={inlier_count}"
    else:
        text = (
            f"omega={angular_velocity:+.3f} deg/frame "
            f"({angular_velocity * fps:+.1f} deg/s, {angular_velocity * fps / 6:+.1f} rpm) "
            f"points={inlier_count}"
        )
    cv2.putText(
        frame,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_video(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    video_writer = make_video_writer(cap, args.output_video)
    csv_file = args.output_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame",
        "omega_deg_per_frame",
        "omega_deg_per_s",
        "rpm",
        "tracked_points",
        "angular_inliers",
        "pose_refreshed",
        "person_mask_used",
    ])

    landmarker, delegate = create_pose_landmarker(output_segmentation_masks=True)
    landmark_indices = FOOT_LANDMARKS[args.side]
    bb_center = args.bb_x, args.bb_y
    previous_gray = None
    tracked_points = None
    foot_landmarks = None
    feature_mask = None
    person_mask_used = False
    frame_index = 0

    print(f"video: {args.video}")
    print(f"BB center: ({args.bb_x:.2f}, {args.bb_y:.2f})")
    print(f"side: {args.side}")
    print(f"delegate: {delegate}")
    print(f"output CSV: {args.output_csv}")
    print(f"output video: {args.output_video}")

    try:
        with landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                angular_velocity = None
                inlier_count = 0
                pose_refreshed = False

                if previous_gray is not None and tracked_points is not None:
                    previous_valid, current_valid, _ = track_features(
                        previous_gray,
                        gray,
                        tracked_points,
                        args.max_forward_backward_error,
                    )
                    angular_velocities = point_angular_velocities(
                        previous_valid,
                        current_valid,
                        bb_center,
                        args.min_flow,
                        args.min_radius,
                        args.min_abs_deg_per_frame,
                        args.max_abs_deg_per_frame,
                    )
                    angular_velocity, inlier_count = robust_angular_velocity(
                        angular_velocities,
                        args.mad_scale,
                    )
                    tracked_points = (
                        current_valid.reshape(-1, 1, 2).astype(np.float32)
                        if len(current_valid) > 0
                        else None
                    )

                needs_pose = (
                    frame_index % args.detect_every == 0
                    or tracked_points is None
                    or len(tracked_points) < args.min_points
                )
                if needs_pose:
                    foot_landmarks, person_mask = detect_foot_landmarks(
                        landmarker,
                        frame,
                        video_timestamp_ms(cap, frame_index),
                        landmark_indices,
                        args.pose_scale,
                        args.person_mask_threshold,
                    )
                    pose_refreshed = True
                    if foot_landmarks is not None:
                        foot_mask = make_foot_mask(gray.shape, foot_landmarks, args.roi_padding)
                        person_mask_used = person_mask is not None
                        feature_mask = (
                            cv2.bitwise_and(foot_mask, person_mask)
                            if person_mask is not None
                            else foot_mask
                        )
                        tracked_points = detect_features(
                            gray,
                            feature_mask,
                            args.max_corners,
                            args.quality_level,
                            args.min_distance,
                        )
                        if tracked_points is not None and len(tracked_points) < args.min_points:
                            tracked_points = None
                    else:
                        feature_mask = None
                        person_mask_used = False
                        tracked_points = None

                tracked_count = 0 if tracked_points is None else len(tracked_points)
                csv_writer.writerow([
                    frame_index,
                    "" if angular_velocity is None else f"{angular_velocity:.9f}",
                    "" if angular_velocity is None else f"{angular_velocity * fps:.9f}",
                    "" if angular_velocity is None else f"{angular_velocity * fps / 6:.9f}",
                    tracked_count,
                    inlier_count,
                    int(pose_refreshed),
                    int(person_mask_used),
                ])
                draw_overlay(
                    frame,
                    bb_center,
                    foot_landmarks,
                    feature_mask,
                    tracked_points,
                    angular_velocity,
                    inlier_count,
                    fps,
                    args.red_at_deg_per_frame,
                )
                video_writer.write(frame)

                if frame_index % args.progress_every == 0:
                    print(
                        f"frame {frame_index}/{total_frames} "
                        f"points={tracked_count} inliers={inlier_count} omega={angular_velocity}"
                    )
                previous_gray = gray
                frame_index += 1
    finally:
        cap.release()
        video_writer.release()
        csv_file.close()

    print(f"saved: {args.output_csv}")
    print(f"saved: {args.output_video}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate BB-centered angular velocity from sparse optical flow on the foot"
    )
    parser.add_argument("--video", type=Path, help="input video; GUI selection opens when omitted")
    parser.add_argument("--side", choices=sorted(FOOT_LANDMARKS), default="left")
    parser.add_argument("--bb-x", type=float, help="BB center x coordinate in pixels")
    parser.add_argument("--bb-y", type=float, help="BB center y coordinate in pixels")
    parser.add_argument(
        "--bb-frame",
        type=int,
        default=0,
        help="video frame used to select BB center by clicking",
    )
    parser.add_argument("--output-csv", type=Path, help="output CSV path")
    parser.add_argument("--output-video", type=Path, help="annotated MP4 path")
    parser.add_argument("--cpu", action="store_true", help="kept for compatibility; CPU is always used")
    parser.add_argument("--detect-every", type=int, default=15, help="refresh foot ROI every N frames")
    parser.add_argument("--pose-scale", type=float, default=0.5, help="Pose detection scale")
    parser.add_argument(
        "--person-mask-threshold",
        type=float,
        default=0.5,
        help="minimum MediaPipe person-segmentation probability used for feature detection",
    )
    parser.add_argument("--roi-padding", type=int, default=35, help="foot ROI padding in pixels")
    parser.add_argument("--max-corners", type=int, default=80, help="maximum sparse feature points")
    parser.add_argument("--min-points", type=int, default=12, help="refresh ROI below this point count")
    parser.add_argument("--quality-level", type=float, default=0.01, help="Shi-Tomasi quality level")
    parser.add_argument("--min-distance", type=float, default=6.0, help="minimum feature distance")
    parser.add_argument(
        "--max-forward-backward-error",
        type=float,
        default=1.5,
        help="maximum forward-backward optical-flow error in pixels",
    )
    parser.add_argument(
        "--min-flow",
        type=float,
        default=0.25,
        help="ignore nearly stationary points below this displacement in pixels/frame",
    )
    parser.add_argument("--min-radius", type=float, default=30.0, help="ignore points this close to BB")
    parser.add_argument(
        "--min-abs-deg-per-frame",
        type=float,
        default=0.5,
        help="reject nearly stationary equipment points below this absolute angular velocity",
    )
    parser.add_argument(
        "--max-abs-deg-per-frame",
        type=float,
        default=15.0,
        help="reject per-point angular velocities above this absolute value",
    )
    parser.add_argument("--mad-scale", type=float, default=3.5, help="MAD outlier rejection scale")
    parser.add_argument(
        "--red-at-deg-per-frame",
        type=float,
        default=5.0,
        help="point color becomes fully red at this absolute angular velocity",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)

    if args.video is None:
        args.video = select_video()
    has_bb_coordinates = args.bb_x is not None or args.bb_y is not None
    if has_bb_coordinates and (args.bb_x is None or args.bb_y is None):
        parser.error("--bb-x and --bb-y must be specified together")
    if has_bb_coordinates:
        if not math.isfinite(args.bb_x) or not math.isfinite(args.bb_y): # type: ignore
            parser.error("--bb-x and --bb-y must be finite")
    else:
        if args.bb_frame < 0:
            parser.error("--bb-frame must be >= 0")
        args.bb_x, args.bb_y = select_bb_center_from_frame(args.video, args.bb_frame)

    if args.output_csv is None:
        args.output_csv = Path("results") / f"{args.video.stem}_{args.side}_foot_sparse_flow.csv"
    if args.output_video is None:
        args.output_video = Path("results") / f"{args.video.stem}_{args.side}_foot_sparse_flow.mp4"

    positive_values = (
        "detect_every",
        "roi_padding",
        "max_corners",
        "min_points",
        "quality_level",
        "min_distance",
        "max_forward_backward_error",
        "min_flow",
        "min_radius",
        "min_abs_deg_per_frame",
        "max_abs_deg_per_frame",
        "mad_scale",
        "red_at_deg_per_frame",
        "progress_every",
    )
    for name in positive_values:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.min_points > args.max_corners:
        parser.error("--min-points must be <= --max-corners")
    if args.min_abs_deg_per_frame >= args.max_abs_deg_per_frame:
        parser.error("--min-abs-deg-per-frame must be less than --max-abs-deg-per-frame")
    if not (0.1 <= args.pose_scale <= 1.0):
        parser.error("--pose-scale must be between 0.1 and 1.0")
    if not (0.0 < args.person_mask_threshold < 1.0):
        parser.error("--person-mask-threshold must be between 0 and 1")
    return args


def main(argv: list[str]) -> int:
    process_video(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
