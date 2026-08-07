#!/usr/bin/env python3
"""Export toe x/y CSV files using MediaPipe Pose on every video frame.

This version uses MediaPipe Tasks (0.10.35) with the Pose Landmarker CPU
delegate. Every input frame is decoded with OpenCV, converted to SRGBA, and
passed to the Tasks API in VIDEO mode without optical-flow tracking.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe_helper import create_pose_landmarker, ensure_pose_model_file
from video_io import video_timestamp_ms

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# PoseLandmarker の結果から「左足先」の座標だけを取り出して CSV に出す。
LEFT_FOOT_INDEX = 31


def extract_left_foot_xy(result: vision.PoseLandmarkerResult, width: float, height: float): # type: ignore
	# pose_landmarks が空なら検出なし。
	# 1 人分のランドマーク配列のうち、LEFT_FOOT_INDEX だけ取り出す。
	if not result.pose_landmarks:
		return None
	landmark = result.pose_landmarks[0][LEFT_FOOT_INDEX]
	# MediaPipe の正規化座標をピクセル座標に戻す。
	return landmark.x * width, landmark.y * height


def output_csv_path(video_path: str) -> Path:
	base = Path(video_path).stem or "video"
	safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", base)
	safe_stem = safe_stem.strip() or "video"
	return RESULTS_DIR / f"{safe_stem}_toe_mp.csv"


def process_video(video_path: str, landmarker: vision.PoseLandmarker, delegate: str) -> None: # type: ignore
	# 1 本の動画を読み、全フレームを MediaPipe Pose で処理して左足先座標を CSV に書き出す。
	print(f"Processing: {video_path}")
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"  -> cannot open: {video_path}")
		return

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	csv_path = output_csv_path(video_path)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	print(f"output CSV: {csv_path.resolve()}")

	with csv_path.open("w", newline="", encoding="utf-8") as f:
		# CSV は frame / x / y の3列だけにする。
		writer = csv.writer(f)
		writer.writerow(["frame", "x", "y"])

		frame_idx = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			# OpenCV の BGR 画像を MediaPipe が扱いやすい RGBA に変換する。
			frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			# Tasks API は mp.Image を受け取るのでここで包む。
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
			timestamp_ms = video_timestamp_ms(cap, frame_idx)
			# VIDEO モードの推論実行。
			result = landmarker.detect_for_video(mp_image, timestamp_ms)
			landmark_xy = extract_left_foot_xy(result, width, height)

			if landmark_xy is None:
				# 検出なしなら空欄で出力する。
				x_value = ""
				y_value = ""
				if frame_idx % 30 == 0:
					print(f"no landmarks at frame={frame_idx}")
			else:
				# 検出ありなら小数6桁で保存する。
				x, y = landmark_xy
				x_value = f"{x:.6f}"
				y_value = f"{y:.6f}"
				if frame_idx % 30 == 0:
					print(f"landmark frame={frame_idx} x={x:.3f} y={y:.3f}")

			# フレーム番号と座標を1行ずつ書く。
			writer.writerow([frame_idx, x_value, y_value])
			print(
				f"now : {frame_idx} / {total_frame} / {x_value} / {y_value} / delegate={delegate} / file {csv_path.name}"
			)

			frame_idx += 1

	cap.release()
	print(f"  -> saved CSV: {csv_path.resolve()}")
	if not csv_path.exists():
		raise RuntimeError(f"CSV was not created: {csv_path}")


def select_videos() -> list[str]:
	# Tk のファイルダイアログで複数動画を選択する。
	# この関数を分けておくと、GUI まわりだけ差し替えやすい。
	try:
		import tkinter as tk
		from tkinter import filedialog
	except Exception as exc:  # pragma: no cover - GUI environment dependent
		raise SystemExit(
			"tkinter が使えません。_tkinter 付きの Python を使ってください。"
			f" ({exc})"
		)

	root = tk.Tk()
	root.withdraw()
	# .mp4 などを複数選べるようにしておく。
	video_files = filedialog.askopenfilenames(
		title="解析する動画を選択",
		filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
	)
	root.destroy()
	return list(video_files)


def parse_args(argv: list[str]) -> argparse.Namespace:
	# --cpu は古い実行コマンドとの互換用。現在は常に CPU delegate を使う。
	# --video は GUI を使わずにコマンドラインで直接動画を渡したいとき用。
	parser = argparse.ArgumentParser(
		description="Export toe x/y from videos using MediaPipe Pose on every frame"
	)
	parser.add_argument("--cpu", action="store_true", help="kept for compatibility; CPU is always used")
	parser.add_argument(
		"--video",
		action="append",
		help="video file path to process (repeatable). If omitted, a tkinter file picker opens.",
	)
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	# 実行の流れは、引数解析 -> 動画選択 -> 1 回だけ landmarker を初期化 -> 各動画を処理、の順。
	args = parse_args(argv)
	RESULTS_DIR.mkdir(exist_ok=True)
	videos = args.video if args.video else select_videos()
	if not videos:
		print("No files selected. 終了します。")
		return 0

	# 1 回初期化した landmarker を全動画で使い回す。
	landmarker, delegate = create_pose_landmarker()
	with landmarker:
		print(f"mediapipe version: {getattr(mp, '__version__', 'unknown')}")
		print(f"delegate: {delegate}")
		print(f"model: {ensure_pose_model_file()}")
		# 選択された動画を順番に処理して CSV を作る。
		for video_path in videos:
			process_video(video_path, landmarker=landmarker, delegate=delegate)

	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
