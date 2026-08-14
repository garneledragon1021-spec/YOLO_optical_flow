#!/usr/bin/env python3
#toe_live.py
"""RTMPose WholeBodyでつま先を検出し、光学フローで追跡するプログラム。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

try:
    import cv2
    import numpy as np
    from rtmlib import Wholebody
except ModuleNotFoundError as exc:
    print("依存関係が不足しています。次のコマンドで環境を整えてから再実行してください。", file=sys.stderr)
    print("  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

# COCO-WholeBodyの足キーポイント。body(17点)の後に左右の足(各3点)が続く。
TOE_LANDMARKS = {
    "left": 17,   # left_big_toe
    "right": 20,  # right_big_toe
}


def compute_preview_shape(frame_shape: tuple[int, int], max_width: int = 960, max_height: int = 540) -> tuple[int, int]:
    """表示用に大きすぎるフレームを縮小し、画面の 1/4 程度に収まるサイズを返す。"""
    height, width = frame_shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return width, height
    return round(width * scale), round(height * scale)


def detect_toe(
    model: Wholebody,
    frame: np.ndarray,
    landmark_index: int,
    confidence: float,
    imgsz: int,
    device: str | None = None,
) -> tuple[float, float] | None:
    """RTMPose/RTMW WholeBodyの133点出力から指定側のつま先を取得する。"""
    keypoints, scores = model(frame)
    xy = np.asarray(keypoints, dtype=float)
    conf_np = np.asarray(scores, dtype=float)
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] <= landmark_index:
        return None
    if conf_np.ndim != 2 or conf_np.shape != xy.shape[:2]:
        conf_np = np.ones(xy.shape[:2])
    valid = conf_np[:, landmark_index] >= confidence
    if not np.any(valid):
        return None
    person_index = int(np.argmax(np.where(valid, conf_np[:, landmark_index], -1.0)))
    x, y = xy[person_index, landmark_index]
    return float(x), float(y)


def crop_bounds(point: tuple[float, float], width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    """点を中心とする追跡クロップを画像範囲内に切り詰めて返す。"""
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
    """前フレームの点を Lucas-Kanade 法で現フレームへ移動させる。"""
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
        None, # type: ignore
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    ) # type: ignore
    if status is None or status[0][0] != 1:
        return None

    next_x = float(next_points[0][0][0] + x1)
    next_y = float(next_points[0][0][1] + y1)
    if not (0 <= next_x < width and 0 <= next_y < height):
        return None
    return next_x, next_y


def timestamp_for_frame(cap: cv2.VideoCapture, frame_idx: int, start_time: float) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return int((frame_idx / fps) * 1000)
    return int((time.monotonic() - start_time) * 1000)


def draw_overlay(
    frame: np.ndarray,
    point: tuple[float, float] | None,
    trail: list[tuple[float, float]],
    crop_size: int,
    delegate: str,
    source: str,
) -> None:
    if len(trail) >= 2:
        points = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], isClosed=False, color=(0, 200, 255), thickness=3)
        for index, (x, y) in enumerate(trail[:: max(1, len(trail) // 12)]):
            alpha = (index + 1) / max(1, len(trail) // 12 + 1)
            cv2.circle(frame, (int(x), int(y)), max(2, int(5 * alpha)), (0, 120, 255), -1)

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


def run(args: argparse.Namespace) -> int:
    cap_source: int | str = args.camera if args.video is None else args.video
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise SystemExit(f"入力を開けませんでした: {cap_source}")

    csv_file = None
    writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "x", "y", "source"])

    model = Wholebody(
        mode=args.rtmpose_mode,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )
    delegate = f"RTMPose WholeBody ({args.rtmpose_mode}, {args.backend}, {args.device})"
    landmark_index = TOE_LANDMARKS[args.side]
    prev_gray = None
    point = None
    trail: deque[tuple[float, float]] = deque(maxlen=args.trail_length)
    source = "none"
    start_time = time.monotonic()
    frame_idx = 0
    preview_window_name = "Realtime Toe Tracker"

    try:
        while True:
                ret, frame = cap.read()
                if not ret:
                    break

                preview_width, preview_height = compute_preview_shape(frame.shape[:2])
                preview = cv2.resize(
                    frame,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                needs_pose = point is None or frame_idx % args.detect_every == 0

                if not needs_pose and prev_gray is not None:
                    tracked = track_toe(prev_gray, gray, point, args.crop_size) # type: ignore
                    if tracked is not None:
                        point = tracked
                        trail.append(point)
                        source = "track"
                    else:
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
                        point = detected
                        trail.append(point)
                        source = "pose"
                    else:
                        point = None
                        trail.clear()
                        source = "none"

                if writer is not None:
                    if point is None:
                        writer.writerow([frame_idx, "", "", source])
                    else:
                        writer.writerow([frame_idx, f"{point[0]:.6f}", f"{point[1]:.6f}", source])

                draw_overlay(frame, point, list(trail), args.crop_size, delegate, source)
                preview = cv2.resize(
                    frame,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(preview_window_name, preview_width, preview_height)
                cv2.imshow(preview_window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                prev_gray = gray
                frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime RTMPose wholebody toe tracker")
    parser.add_argument("--video", help="video file path. If omitted, the camera is used.")
    parser.add_argument("--camera", type=int, default=0, help="camera index when --video is omitted")
    parser.add_argument("--side", choices=sorted(TOE_LANDMARKS), default="left", help="toe landmark to track")
    parser.add_argument("--cpu", action="store_true", help="use CPU inference")
    parser.add_argument("--device", default="cpu", help="推論デバイス。例: cpu, cuda, mps")
    parser.add_argument(
        "--backend",
        choices=["onnxruntime", "opencv", "openvino"],
        default="onnxruntime",
        help="RTMPoseの推論バックエンド",
    )
    parser.add_argument(
        "--rtmpose-mode",
        choices=["performance", "balanced", "lightweight"],
        default="lightweight",
        help="WholeBodyモデル。初回実行時に公式ONNXモデルを自動取得する",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="keypoint/person confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="互換用オプション（RTMPoseではモデル設定を使用）")
    parser.add_argument("--detect-every", type=int, default=30, help="run RTMPose every N frames")
    parser.add_argument("--crop-size", type=int, default=160, help="tracking crop size around the toe in pixels")
    parser.add_argument("--trail-length", type=int, default=90, help="number of recent points to display as a trajectory")
    parser.add_argument("--csv", help="optional CSV output path")
    args = parser.parse_args(argv)
    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.crop_size < 32:
        parser.error("--crop-size must be >= 32")
    if not (0.0 < args.confidence <= 1.0):
        parser.error("--confidence must be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be >= 32")
    if args.trail_length < 0:
        parser.error("--trail-length must be >= 0")
    if args.cpu:
        args.device = "cpu"
    return args


def main(argv: list[str]) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
