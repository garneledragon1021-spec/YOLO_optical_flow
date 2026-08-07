#!/usr/bin/env python3
"""HSV 色分離で蛍光マーカーを検出し、座標 CSV と注釈付き動画を出力する。

色マスクから輪郭を抽出し、面積・円形度・縦横比・前フレームからの距離を
使って候補を絞る。赤い BB と黄色い靴を同時に扱うモードにも対応している。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_io import make_video_writer


RESULTS_DIR = Path("results")
HSVRange = tuple[tuple[int, int, int], tuple[int, int, int]]


COLOR_PRESETS: dict[str, list[HSVRange]] = {
    "green": [((35, 80, 80), (95, 255, 255))],
    "yellow": [((23, 45, 55), (58, 255, 255))],
    "orange": [((0, 50, 50), (22, 255, 255))],
    "pink": [((140, 60, 80), (179, 255, 255))],
    "red": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (179, 255, 255))],
}
BB_SHOE_COLORS = {
    "bb": "red",
    "shoe": "yellow",
}
BB_SHOE_LABELS = {
    "bb": "BB",
    "shoe": "shoe",
}
BB_SHOE_DRAW_COLORS = {
    "bb": (0, 140, 255),
    "shoe": (0, 255, 255),
}


def velocity_color(speed: float | None, max_speed: float = 20.0) -> tuple[int, int, int]:
    """速度を緑→黄→赤へ変換する（OpenCVのBGR）。"""
    if speed is None or not math.isfinite(speed):
        return (255, 255, 255)
    ratio = min(max(abs(speed) / max(max_speed, 1e-9), 0.0), 1.0)
    if ratio < 0.5:
        t = ratio * 2.0
        return (0, 255, round(255 * t))
    t = (ratio - 0.5) * 2.0
    return (0, round(255 * (1.0 - t)), 255)


@dataclass(frozen=True)
class MarkerDetection:
    x: float
    y: float
    area: float
    radius: float
    bbox: tuple[int, int, int, int]


def select_video() -> Path:
    """GUIで入力動画を1本選択する。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - GUI environment dependent
        raise SystemExit(f"tkinter is unavailable; specify --video ({exc})")

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="マーカ座標を取得する動画を選択",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    if not selected:
        raise SystemExit("動画選択がキャンセルされました")
    return Path(selected)


def select_point_from_frame(video_path: Path, frame_index: int) -> tuple[float, float]:
    """指定フレームを表示し、クリックしたBB中心座標を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"BB選択用の動画を開けませんでした: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"--bb-frame {frame_index} を読み込めませんでした")

    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 800 / height)
    preview = (
        cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else frame
    )
    selected: list[tuple[float, float]] = []
    window_name = f"Click BB center - frame {frame_index} (Enter: confirm, Esc: cancel)"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        """クリック位置を元動画の座標として保存する。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(x / scale, y / scale)]

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            display = preview.copy()
            if selected:
                point = round(selected[0][0] * scale), round(selected[0][1] * scale)
                cv2.drawMarker(
                    display,
                    point,
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
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
                raise SystemExit("BB中心の選択がキャンセルされました")
    finally:
        cv2.destroyWindow(window_name)


def parse_hsv(value: str) -> tuple[int, int, int]:
    """H,S,V 形式の文字列を OpenCV の HSV 範囲として検証する。"""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV must be H,S,V, for example 35,80,80")
    try:
        h, s, v = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc
    if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
        raise argparse.ArgumentTypeError("HSV range is H=0..179, S=0..255, V=0..255")
    return h, s, v


def hsv_ranges_from_args(args: argparse.Namespace) -> list[HSVRange]:
    """コマンドライン引数からHSV抽出範囲を作る。"""
    if args.color == "bb_shoe":
        raise SystemExit("--lower-hsv/--upper-hsv cannot be used with --color bb_shoe")
    if args.lower_hsv is None and args.upper_hsv is None:
        return COLOR_PRESETS[args.color]
    if args.lower_hsv is None or args.upper_hsv is None:
        raise SystemExit("--lower-hsv and --upper-hsv must be specified together")
    lower = args.lower_hsv
    upper = args.upper_hsv
    if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
        raise SystemExit("--lower-hsv values must be <= --upper-hsv values")
    return [(lower, upper)]


def build_marker_mask(
    frame: np.ndarray,
    hsv_ranges: list[HSVRange],
    blur_kernel: int,
    morph_kernel: int,
) -> np.ndarray:
    """フレームから指定色の二値マスクを作る。"""
    """フレームを HSV に変換し、指定色に該当する二値マスクを作る。

    Gaussian blur は細かな色ノイズを抑え、open/close は点状ノイズ除去と
    マーカー領域の穴埋めを行う。
    """
    if blur_kernel > 1:
        frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        lower_array = np.asarray(lower, dtype=np.uint8)
        upper_array = np.asarray(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_array, upper_array))

    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def contour_center(contour: np.ndarray) -> tuple[float, float]:
    """輪郭の重心または外接円中心を返す。"""
    moments = cv2.moments(contour)
    if moments["m00"]:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    (circle_x, circle_y), _ = cv2.minEnclosingCircle(contour)
    return float(circle_x), float(circle_y)


def contour_circularity(contour: np.ndarray, area: float) -> float:
    """輪郭の円形度を 0〜1 付近で計算する（円に近いほど 1）。"""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def contour_aspect_score(contour: np.ndarray) -> float:
    """輪郭の外接矩形が正方形に近いほど高い値を返す。"""
    _, _, width, height = cv2.boundingRect(contour)
    longest = max(width, height)
    if longest <= 0:
        return 0.0
    return min(width, height) / longest


def detect_marker(
    mask: np.ndarray,
    min_area: float,
    max_area: float | None,
    *,
    prefer_circular: bool = False,
    target_area: float | None = None,
    reference_point: tuple[float, float] | None = None,
    max_reference_distance: float | None = None,
    min_circularity: float = 0.0,
    min_aspect: float = 0.0,
) -> MarkerDetection | None:
    """マスク内の条件に合う輪郭を1つ検出する。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        x, y = contour_center(contour)
        distance = 0.0
        if reference_point is not None:
            distance = math.hypot(x - reference_point[0], y - reference_point[1])
            if max_reference_distance is not None and distance > max_reference_distance:
                continue
        if prefer_circular:
            circularity = contour_circularity(contour, area)
            aspect = contour_aspect_score(contour)
            if circularity < min_circularity or aspect < min_aspect:
                continue
            area_score = 1.0
            if target_area is not None and target_area > 0:
                area_score = min(area, target_area) / max(area, target_area)
            score = circularity * aspect * area_score
            if reference_point is not None and max_reference_distance is not None:
                distance_score = max(1.0 - distance / max_reference_distance, 0.0)
                score *= distance_score * distance_score
            candidates.append((score, area, contour))
        else:
            candidates.append((area, area, contour))
    if not candidates:
        return None

    _, area, contour = max(candidates, key=lambda item: item[0])
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
    x, y = contour_center(contour)
    return MarkerDetection(x=x, y=y, area=area, radius=float(radius), bbox=cv2.boundingRect(contour))


def offset_detection(
    detection: MarkerDetection | None,
    x_offset: int,
    y_offset: int,
) -> MarkerDetection | None:
    """検出結果の座標と矩形をROI分だけ移動する。"""
    if detection is None:
        return None
    x, y, width, height = detection.bbox
    return MarkerDetection(
        x=detection.x + x_offset,
        y=detection.y + y_offset,
        area=detection.area,
        radius=detection.radius,
        bbox=(x + x_offset, y + y_offset, width, height),
    )


def detect_bb_near_anchor(
    frame: np.ndarray,
    anchor: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[MarkerDetection | None, np.ndarray]:
    """BB中心付近だけを探索して赤マーカーを検出する。"""
    height, width = frame.shape[:2]
    radius = math.ceil(args.bb_search_radius)
    x0 = max(math.floor(anchor[0]) - radius, 0)
    y0 = max(math.floor(anchor[1]) - radius, 0)
    x1 = min(math.floor(anchor[0]) + radius + 1, width)
    y1 = min(math.floor(anchor[1]) + radius + 1, height)
    roi = frame[y0:y1, x0:x1]
    roi_mask = build_marker_mask(
        roi,
        COLOR_PRESETS[BB_SHOE_COLORS["bb"]],
        args.blur_kernel,
        args.morph_kernel,
    )
    detection = detect_marker(
        roi_mask,
        args.min_area,
        args.bb_max_area,
        prefer_circular=True,
        target_area=args.bb_target_area,
        reference_point=(anchor[0] - x0, anchor[1] - y0),
        max_reference_distance=args.bb_max_motion,
        min_circularity=args.bb_min_circularity,
        min_aspect=args.bb_min_aspect,
    )
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = roi_mask
    return offset_detection(detection, x0, y0), full_mask


def draw_status_line(
    frame: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """動画上に黒縁付きの状態テキストを描く。"""
    origin = (12, y)
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_detection_label(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    *,
    prefer_below: bool = False,
) -> None:
    """検出マーカーの近くに説明ラベルを描く。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_size, baseline = cv2.getTextSize(text, font, font_scale, 1)
    text_width, text_height = text_size
    x = min(max(center[0] + radius + 6, 4), max(frame.shape[1] - text_width - 4, 4))
    above_y = center[1] - radius - 8
    below_y = center[1] + radius + text_height + 8
    below_fits = below_y <= frame.shape[0] - baseline - 4
    if prefer_below and below_fits:
        y = below_y
    elif above_y >= text_height:
        y = above_y
    else:
        y = min(below_y, frame.shape[0] - baseline - 4)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detection: MarkerDetection | None,
    mask: np.ndarray | None,
    frame_index: int,
    show_mask: bool,
    speed: float | None = None,
    speed_color_max: float = 20.0,
) -> np.ndarray:
    """単一マーカーの検出結果をフレームへ重ねる。"""
    output = frame.copy()
    if show_mask and mask is not None:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(output, 0.75, mask_bgr, 0.25, 0)

    if detection is not None:
        x, y, w, h = detection.bbox
        center = round(detection.x), round(detection.y)
        radius = max(round(detection.radius), 3)
        color = velocity_color(speed, speed_color_max)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        draw_detection_label(
            output,
            f"marker ({detection.x:.1f}, {detection.y:.1f})"
            + (f" speed={speed:.1f}" if speed is not None else ""),
            center,
            radius,
            color,
        )

    draw_status_line(output, f"frame={frame_index}", 24)
    if detection is None:
        draw_status_line(output, "marker: not detected", 48)
    return output


def draw_multi_overlay(
    frame: np.ndarray,
    detections: dict[str, MarkerDetection | None],
    masks: dict[str, np.ndarray],
    frame_index: int,
    show_mask: bool,
    estimated_targets: set[str] | None = None,
    shoe_speed: float | None = None,
    speed_color_max: float = 20.0,
) -> np.ndarray:
    """BBと靴マーカーの検出結果を速度色付きで重ねる。"""
    output = frame.copy()
    estimated_targets = estimated_targets or set()
    if show_mask:
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for mask in masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(output, 0.75, mask_bgr, 0.25, 0)

    for target, detection in detections.items():
        label = BB_SHOE_LABELS[target]
        estimated = target in estimated_targets
        color = (255, 255, 255) if estimated else BB_SHOE_DRAW_COLORS[target]
        if target == "shoe":
            color = velocity_color(shoe_speed, speed_color_max)
        if detection is None:
            continue

        x, y, w, h = detection.bbox
        center = round(detection.x), round(detection.y)
        radius = max(round(detection.radius), 3)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.circle(output, center, radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        draw_detection_label(
            output,
            f"{label}{' estimated' if estimated else ''} ({detection.x:.1f}, {detection.y:.1f})"
            + (f" speed={shoe_speed:.1f}" if target == "shoe" and shoe_speed is not None else ""),
            center,
            radius,
            color,
            prefer_below=target == "shoe",
        )

    draw_status_line(output, f"frame={frame_index}", 24)
    return output


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """入力名からCSVと動画の既定出力先を決める。"""
    if args.video is not None:
        stem = args.video.stem
    else:
        stem = f"camera{args.camera}"
    csv_path = args.csv if args.csv is not None else RESULTS_DIR / f"{stem}_marker.csv"
    video_path = args.output_video
    if args.write_video and video_path is None:
        video_path = RESULTS_DIR / f"{stem}_marker.mp4"
    return csv_path, video_path


def estimated_marker(point: tuple[float, float], target_area: float) -> MarkerDetection:
    """最後に分かった位置から推定マーカー情報を作る。"""
    radius = max(math.sqrt(target_area / math.pi), 3.0)
    x0 = round(point[0] - radius)
    y0 = round(point[1] - radius)
    diameter = max(round(radius * 2), 1)
    return MarkerDetection(
        x=point[0],
        y=point[1],
        area=0.0,
        radius=radius,
        bbox=(x0, y0, diameter, diameter),
    )


def process(args: argparse.Namespace) -> None:
    """動画を走査して検出結果CSVと注釈動画を書き出す。"""
    source: int | str
    if args.video is not None:
        source = str(args.video)
    else:
        source = args.camera

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"入力を開けませんでした: {source}")

    hsv_ranges = None if args.color == "bb_shoe" else hsv_ranges_from_args(args)
    csv_path, output_video_path = default_output_paths(args)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = None
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = make_video_writer(cap, output_video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bb_anchor = (
        (float(args.bb_x), float(args.bb_y))
        if args.bb_x is not None and args.bb_y is not None
        else None
    )
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if bb_anchor is not None and not (
        0 <= bb_anchor[0] < frame_width and 0 <= bb_anchor[1] < frame_height
    ):
        cap.release()
        raise SystemExit(
            f"BB anchor {bb_anchor} is outside the video frame {frame_width}x{frame_height}"
        )
    print(f"input: {source}")
    if args.color == "bb_shoe":
        print("HSV ranges:")
        for target, color in BB_SHOE_COLORS.items():
            print(f"  {BB_SHOE_LABELS[target]} ({color}): {COLOR_PRESETS[color]}")
        if bb_anchor is not None:
            print(f"BB anchor: ({bb_anchor[0]:.1f}, {bb_anchor[1]:.1f})")
    else:
        print(f"HSV ranges: {hsv_ranges}")
    print(f"output CSV: {csv_path}")
    if output_video_path is not None:
        print(f"output video: {output_video_path}")

    frame_index = 0
    detection_counts = (
        {target: 0 for target in BB_SHOE_COLORS}
        if args.color == "bb_shoe"
        else {args.color: 0}
    )
    previous_detections: dict[str, MarkerDetection | None] = {target: None for target in BB_SHOE_COLORS}
    bb_last_point = bb_anchor
    bb_estimated_count = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            previous_shoe_point: tuple[float, float] | None = None
            if args.color == "bb_shoe":
                writer.writerow([
                    "frame",
                    "bb_x",
                    "bb_y",
                    "bb_area",
                    "bb_radius",
                    "bb_bbox_x",
                    "bb_bbox_y",
                    "bb_bbox_w",
                    "bb_bbox_h",
                    "bb_detected",
                    "bb_estimated",
                    "shoe_x",
                    "shoe_y",
                    "shoe_area",
                    "shoe_radius",
                    "shoe_bbox_x",
                    "shoe_bbox_y",
                    "shoe_bbox_w",
                    "shoe_bbox_h",
                    "shoe_detected",
                    "shoe_speed_px_per_frame",
                ])
            else:
                writer.writerow([
                    "frame",
                    "x",
                    "y",
                    "area",
                    "radius",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                    "detected",
                    "speed_px_per_frame",
                ])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.color == "bb_shoe":
                    if bb_anchor is not None:
                        bb_detection, bb_mask = detect_bb_near_anchor(frame, bb_anchor, args)
                    else:
                        bb_mask = build_marker_mask(
                            frame,
                            COLOR_PRESETS[BB_SHOE_COLORS["bb"]],
                            args.blur_kernel,
                            args.morph_kernel,
                        )
                        previous_bb = previous_detections["bb"]
                        bb_detection = detect_marker(
                            bb_mask,
                            args.min_area,
                            args.bb_max_area,
                            prefer_circular=True,
                            target_area=args.bb_target_area,
                            reference_point=(previous_bb.x, previous_bb.y) if previous_bb is not None else None,
                            max_reference_distance=args.bb_max_motion,
                            min_circularity=args.bb_min_circularity,
                            min_aspect=args.bb_min_aspect,
                        )

                    shoe_mask = build_marker_mask(
                        frame,
                        COLOR_PRESETS[BB_SHOE_COLORS["shoe"]],
                        args.blur_kernel,
                        args.morph_kernel,
                    )
                    shoe_detection = detect_marker(shoe_mask, args.min_area, args.max_area)
                    raw_detections = {"bb": bb_detection, "shoe": shoe_detection}
                    for target, detection in raw_detections.items():
                        if detection is not None:
                            previous_detections[target] = detection
                            detection_counts[target] += 1
                        elif target == "bb" and bb_anchor is None:
                            previous_detections[target] = None

                    estimated_targets: set[str] = set()
                    output_bb = bb_detection
                    if bb_detection is not None:
                        bb_last_point = (bb_detection.x, bb_detection.y)
                    elif bb_last_point is not None:
                        output_bb = estimated_marker(bb_last_point, args.bb_target_area)
                        estimated_targets.add("bb")
                        bb_estimated_count += 1
                    output_detections = {"bb": output_bb, "shoe": shoe_detection}
                    shoe_speed = None
                    if shoe_detection is not None and previous_shoe_point is not None:
                        shoe_speed = math.hypot(
                            shoe_detection.x - previous_shoe_point[0],
                            shoe_detection.y - previous_shoe_point[1],
                        )
                    if shoe_detection is not None:
                        previous_shoe_point = (shoe_detection.x, shoe_detection.y)

                    row = [frame_index]
                    if output_bb is None:
                        row.extend(["", "", "", "", "", "", "", "", 0, 0])
                    elif bb_detection is None:
                        row.extend([
                            f"{output_bb.x:.6f}",
                            f"{output_bb.y:.6f}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            0,
                            1,
                        ])
                    else:
                        row.extend([
                            f"{bb_detection.x:.6f}",
                            f"{bb_detection.y:.6f}",
                            f"{bb_detection.area:.6f}",
                            f"{bb_detection.radius:.6f}",
                            *bb_detection.bbox,
                            1,
                            0,
                        ])
                    if shoe_detection is None:
                        row.extend(["", "", "", "", "", "", "", "", 0])
                    else:
                        row.extend([
                            f"{shoe_detection.x:.6f}",
                            f"{shoe_detection.y:.6f}",
                            f"{shoe_detection.area:.6f}",
                            f"{shoe_detection.radius:.6f}",
                            *shoe_detection.bbox,
                            1,
                        ])
                    row.append("" if shoe_speed is None else f"{shoe_speed:.6f}")
                    writer.writerow(row)

                    if video_writer is not None or args.preview:
                        overlay = draw_multi_overlay(
                            frame,
                            output_detections,
                            {"bb": bb_mask, "shoe": shoe_mask},
                            frame_index,
                            args.show_mask,
                            estimated_targets,
                            shoe_speed,
                            args.speed_color_max,
                        )
                        if video_writer is not None:
                            video_writer.write(overlay)
                        if args.preview:
                            cv2.imshow("Marker Tracker", overlay)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                else:
                    assert hsv_ranges is not None
                    mask = build_marker_mask(frame, hsv_ranges, args.blur_kernel, args.morph_kernel)
                    detection = detect_marker(mask, args.min_area, args.max_area)
                    speed = None
                    if detection is not None and previous_shoe_point is not None:
                        speed = math.hypot(detection.x - previous_shoe_point[0], detection.y - previous_shoe_point[1])
                    if detection is not None:
                        previous_shoe_point = (detection.x, detection.y)
                    if detection is None:
                        writer.writerow([frame_index, "", "", "", "", "", "", "", "", 0, ""])
                    else:
                        detection_counts[args.color] += 1
                        writer.writerow([
                            frame_index,
                            f"{detection.x:.6f}",
                            f"{detection.y:.6f}",
                            f"{detection.area:.6f}",
                            f"{detection.radius:.6f}",
                            *detection.bbox,
                            1,
                            "" if speed is None else f"{speed:.6f}",
                        ])

                    if video_writer is not None or args.preview:
                        overlay = draw_overlay(
                            frame, detection, mask, frame_index, args.show_mask,
                            speed, args.speed_color_max,
                        )
                        if video_writer is not None:
                            video_writer.write(overlay)
                        if args.preview:
                            cv2.imshow("Marker Tracker", overlay)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                if frame_index % args.progress_every == 0:
                    if total_frames > 0:
                        print(f"frame {frame_index}/{total_frames}")
                    else:
                        print(f"frame {frame_index}")
                frame_index += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"saved: {csv_path}")
    if output_video_path is not None:
        print(f"saved: {output_video_path}")
    for target, count in detection_counts.items():
        label = BB_SHOE_LABELS.get(target, target)
        print(f"detected {label}: {count}/{frame_index} frames")
    if args.color == "bb_shoe" and bb_anchor is not None:
        print(f"estimated BB during occlusion: {bb_estimated_count}/{frame_index} frames")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """コマンドライン引数を解析して検証する。"""
    parser = argparse.ArgumentParser(description="Detect fluorescent marker coordinates with OpenCV HSV thresholding")
    parser.add_argument("--video", type=Path, help="input video path; GUI selection opens when omitted")
    parser.add_argument("--camera", type=int, help="camera index. Specify this instead of --video for live capture")
    parser.add_argument(
        "--color",
        choices=sorted([*COLOR_PRESETS, "bb_shoe"]),
        default="bb_shoe",
        help="fluorescent marker color preset; bb_shoe detects red BB and yellow shoe markers",
    )
    parser.add_argument("--lower-hsv", type=parse_hsv, help="custom lower HSV as H,S,V")
    parser.add_argument("--upper-hsv", type=parse_hsv, help="custom upper HSV as H,S,V")
    parser.add_argument("--min-area", type=float, default=10.0, help="minimum marker contour area in pixels")
    parser.add_argument("--max-area", type=float, help="maximum marker contour area in pixels")
    parser.add_argument("--bb-x", type=float, help="approximate BB center x coordinate")
    parser.add_argument("--bb-y", type=float, help="approximate BB center y coordinate")
    parser.add_argument(
        "--bb-frame",
        type=int,
        help="video frame on which to click the BB center; cannot be used with --bb-x/--bb-y",
    )
    parser.add_argument(
        "--bb-search-radius",
        type=float,
        default=80.0,
        help="half-size of the BB search ROI around the selected center",
    )
    parser.add_argument(
        "--bb-max-area",
        type=float,
        default=500.0,
        help="maximum BB marker contour area in bb_shoe mode",
    )
    parser.add_argument(
        "--bb-target-area",
        type=float,
        default=200.0,
        help="expected BB marker contour area used to prefer circular candidates in bb_shoe mode",
    )
    parser.add_argument(
        "--bb-max-motion",
        type=float,
        default=40.0,
        help="maximum distance from the selected BB center in bb_shoe mode",
    )
    parser.add_argument(
        "--bb-min-circularity",
        type=float,
        default=0.6,
        help="minimum BB contour circularity (0..1)",
    )
    parser.add_argument(
        "--bb-min-aspect",
        type=float,
        default=0.6,
        help="minimum BB contour bounding-box aspect ratio (0..1)",
    )
    parser.add_argument("--blur-kernel", type=int, default=5, help="odd Gaussian blur kernel size; 1 disables blur")
    parser.add_argument("--morph-kernel", type=int, default=3, help="morphology kernel size; 1 disables morphology")
    parser.add_argument("--csv", type=Path, help="output CSV path")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write annotated MP4 (default: enabled for video input, disabled for camera input)",
    )
    parser.add_argument("--output-video", type=Path, help="annotated MP4 path")
    parser.add_argument("--preview", action="store_true", help="show OpenCV preview window; press q to stop")
    parser.add_argument("--show-mask", action="store_true", help="blend the HSV mask into preview/output video")
    parser.add_argument("--speed-color-max", type=float, default=20.0, help="speed at which marker becomes red (px/frame)")
    parser.add_argument("--progress-every", type=int, default=100, help="print progress every N frames")
    args = parser.parse_args(argv)

    if args.video is not None and args.camera is not None:
        parser.error("specify either --video or --camera, not both")
    if args.video is None and args.camera is None:
        args.video = select_video()
    if args.video is not None and not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if (args.bb_x is None) != (args.bb_y is None):
        parser.error("--bb-x and --bb-y must be specified together")
    if args.bb_frame is not None and args.bb_x is not None:
        parser.error("--bb-frame cannot be used with --bb-x/--bb-y")
    if args.bb_frame is not None:
        if args.bb_frame < 0:
            parser.error("--bb-frame must be >= 0")
        if args.video is None:
            parser.error("--bb-frame requires --video")
        args.bb_x, args.bb_y = select_point_from_frame(args.video, args.bb_frame)
    if args.color != "bb_shoe" and (args.bb_x is not None or args.bb_frame is not None):
        parser.error("BB selection options require --color bb_shoe")
    if args.color == "bb_shoe" and args.bb_x is None:
        if args.video is None:
            parser.error("--color bb_shoe with a camera requires --bb-x and --bb-y")
        args.bb_frame = 0
        args.bb_x, args.bb_y = select_point_from_frame(args.video, args.bb_frame)
    if args.color == "bb_shoe" and (args.lower_hsv is not None or args.upper_hsv is not None):
        parser.error("--lower-hsv/--upper-hsv cannot be used with --color bb_shoe")
    for name in ("min_area", "progress_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.speed_color_max <= 0 or not math.isfinite(args.speed_color_max):
        parser.error("--speed-color-max must be finite and > 0")
    if args.max_area is not None and args.max_area <= args.min_area:
        parser.error("--max-area must be greater than --min-area")
    if args.bb_max_area <= args.min_area:
        parser.error("--bb-max-area must be greater than --min-area")
    if args.bb_target_area <= 0 or not math.isfinite(args.bb_target_area):
        parser.error("--bb-target-area must be finite and > 0")
    for name in ("bb_search_radius", "bb_max_motion"):
        value = getattr(args, name)
        if value <= 0 or not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    for name in ("bb_min_circularity", "bb_min_aspect"):
        value = getattr(args, name)
        if not 0 <= value <= 1 or not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite and in 0..1")
    if args.bb_x is not None and (not math.isfinite(args.bb_x) or not math.isfinite(args.bb_y)):
        parser.error("--bb-x and --bb-y must be finite")
    for name in ("blur_kernel", "morph_kernel"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            parser.error(f"--{name.replace('_', '-')} must be an odd integer >= 1")
    if args.output_video is not None:
        args.write_video = True
    elif args.write_video is None:
        args.write_video = args.video is not None
    if args.video is not None and args.output_video is not None:
        try:
            if args.video.resolve() == args.output_video.resolve():
                parser.error("--output-video must differ from --video")
        except OSError:
            pass
    if not math.isfinite(args.min_area) or (args.max_area is not None and not math.isfinite(args.max_area)):
        parser.error("--min-area and --max-area must be finite")
    return args


def main(argv: list[str]) -> int:
    """引数解析と動画処理を実行する。"""
    process(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
