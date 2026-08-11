# YOLO Pose + Optical Flow サンプル

OpenCV、Ultralytics YOLO Pose、Lucas–Kanade法によるオプティカルフローを使って、動画またはカメラ映像から足元のキーポイントを検出・追跡するプログラム群です。

## 前提

- Python 3.11系を推奨
- OpenCV
- Ultralytics
- カメラを使う場合はOSのカメラ許可が必要
- YOLOモデルは初回実行時にUltralyticsから自動取得されます

## セットアップ

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install ultralytics
```

実行時は、プロジェクトのルートで仮想環境のPythonを使います。

```powershell
.\.venv\Scripts\python.exe toe_live.py
```

## `toe_live.py`

YOLO Poseを一定間隔で実行し、検出の間のフレームは足元の小さなクロップ内をオプティカルフローで追跡します。検出に失敗した場合も、次のフレームでYOLO Poseに戻ります。

標準のYOLO Poseモデル（COCO形式）にはつま先専用キーポイントがないため、この実装では次の足首キーポイントを追跡します。

| オプション | キーポイント |
| --- | --- |
| `--side left` | 左足首（COCO index 15） |
| `--side right` | 右足首（COCO index 16） |

カメラを使う場合:

```powershell
.\.venv\Scripts\python.exe toe_live.py --side left
```

動画を使う場合:

```powershell
.\.venv\Scripts\python.exe toe_live.py --video path\to\input.mp4 --side left
```

CPUで実行する場合:

```powershell
.\.venv\Scripts\python.exe toe_live.py --video path\to\input.mp4 --cpu
```

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--model` | `yolo11n-pose.pt` | YOLO Poseモデルの名前またはパス |
| `--confidence` | `0.35` | 人物・キーポイントの信頼度しきい値 |
| `--imgsz` | `640` | YOLO推論画像サイズ |
| `--detect-every` | `15` | YOLO Poseを再実行する間隔（フレーム） |
| `--crop-size` | `160` | オプティカルフロー追跡範囲の大きさ（px） |
| `--trail-length` | `90` | 画面に表示する直近軌道の点数 |
| `--device` | 自動 | `cpu`、`0`などの推論デバイス |
| `--csv` | なし | `frame,x,y,source`形式のCSV出力先 |

例:

```powershell
.\.venv\Scripts\python.exe toe_live.py `
  --video path\to\input.mp4 `
  --model yolo11n-pose.pt `
  --side right `
  --detect-every 10 `
  --trail-length 120 `
  --confidence 0.4 `
  --csv results\right_ankle.csv
```

画面表示中に `q` キーを押すと終了します。CSVの `source` は、YOLO Poseによる検出が `pose`、オプティカルフローによる追跡が `track`、未検出が `none` です。

## その他のスクリプト

`toe_mp.py`、`foot_flow.py` などには元のMediaPipe版の処理が残っています。`toe_live.py` と `toe_flow.py` はYOLO Pose版へ置き換え済みです。

| ファイル | 概要 |
| --- | --- |
| `toe_live.py` | YOLO Poseとオプティカルフローによるリアルタイム足首追跡 |
| `toe_flow.py` | YOLO Poseとオプティカルフローによる動画追跡。CSVと任意の注釈付きMP4を出力 |
| `toe_mp.py` | MediaPipe Poseによるキーポイント検出 |
| `trajectory_angle.py` | 座標CSVから軌跡角度・角速度を計算 |
| `overlay_video.py` | 座標CSVを動画へ重ねてMP4を作成 |
| `marker.py` | HSV色抽出によるマーカ検出 |
| `pedal_velocity.py` | BB・ペダル座標から速度やrpmを計算 |

## 注意

標準COCOモデルで本当のつま先位置を取得するには、足首からの補間、専用のキーポイントモデル、または足元を対象に追加学習したYOLO Poseモデルが必要です。
