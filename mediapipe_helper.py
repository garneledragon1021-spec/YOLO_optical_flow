"""MediaPipe Pose Landmarker を CPU で初期化する共通ヘルパー。

各解析スクリプトが個別にモデルをダウンロード・初期化しないように、
モデルの保存場所、CPU delegate、VIDEO モードのオプションをここへ集約する。
"""

from __future__ import annotations

import os
import platform
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOG_LEVEL", "error")

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = MODEL_CACHE_DIR / "pose_landmarker_full.task"


def is_apple_silicon() -> bool:
    """実行環境が Apple Silicon の macOS かどうかを返す。"""
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def cpu_delegate_name() -> str:
    """ログ表示用の CPU delegate 名を返す。"""
    if is_apple_silicon():
        return "CPU (Apple Silicon)"
    return "CPU"


def ensure_model_file(url: str, path: Path) -> Path:
    """モデルをキャッシュへ一度だけ取得し、ローカルパスを返す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading model: {url}")
        urllib.request.urlretrieve(url, path)
    return path


def ensure_pose_model_file() -> Path:
    return ensure_model_file(POSE_MODEL_URL, POSE_MODEL_PATH)


def cpu_base_options(model_path: Path) -> mp_tasks.BaseOptions:
    return mp_tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )


def make_pose_options(
    *,
    output_segmentation_masks: bool = False,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> vision.PoseLandmarkerOptions:
    """Pose Landmarker 用の共通設定を作成する。

    running_mode=VIDEO はフレームの時系列を MediaPipe に伝え、
    frame timestamp が単調増加する限り内部追跡も利用できる設定である。
    """
    return vision.PoseLandmarkerOptions(
        base_options=cpu_base_options(ensure_pose_model_file()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
    )


def create_pose_landmarker(
    *,
    output_segmentation_masks: bool = False,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[vision.PoseLandmarker, str]:
    return (
        vision.PoseLandmarker.create_from_options(
            make_pose_options(
                output_segmentation_masks=output_segmentation_masks,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        ),
        cpu_delegate_name(),
    )
