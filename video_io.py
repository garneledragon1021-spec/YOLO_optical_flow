"""複数の解析スクリプトで共有する OpenCV 動画入出力ヘルパー。"""

from __future__ import annotations

from pathlib import Path

import cv2


def make_video_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    """入力動画と同じサイズ・FPSの MP4 writer を作成する。

    動画によって FPS メタデータが取得できない場合は、保守的に 30 FPS を使う。
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"cannot create output video: {output_path}")
    return writer


def video_timestamp_ms(cap: cv2.VideoCapture, frame_idx: int) -> int:
    """フレーム番号を MediaPipe Tasks が要求するミリ秒 timestamp に変換する。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return int((frame_idx / fps) * 1000)
    return frame_idx * 33
