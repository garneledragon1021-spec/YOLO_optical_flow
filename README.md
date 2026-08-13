# RTMPose WholeBody + Optical Flow サンプル

OpenCV、RTMPose WholeBody、Lucas–Kanade法によるオプティカルフローを使って、動画またはカメラ映像からつま先を検出・追跡するプログラム群です。

## 前提

- Python 3.11系を推奨
- OpenCV
- rtmlib / ONNX Runtime
- カメラを使う場合はOSのカメラ許可が必要
- RTMPose WholeBodyモデルは初回実行時にOpenMMLabから自動取得されます

## セットアップ

Windows PowerShell:

```powershell
cd C:\Users\garne\Github\BYCYCLE\YOLO_optical_flow
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

セットアップを自動化する場合は、次のコマンドでも構築できます。

```powershell
cd C:\Users\garne\Github\BYCYCLE\YOLO_optical_flow
.\setup_venv.ps1
```

PowerShellのスクリプト実行が制限されている場合は、現在のターミナルだけ一時的に許可します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_venv.ps1
```

既存の仮想環境を作り直す場合:

```powershell
.\setup_venv.ps1 -Recreate
```

仮想環境は `YOLO_optical_flow\.venv` に作成されます。実行時は、このフォルダ内のPythonを使います。

```powershell
.\.venv\Scripts\python.exe toe_live-RTMpose.py
```

## `toe_live-RTMpose.py`

RTMPose WholeBodyを一定間隔で実行し、検出の間のフレームは足元の小さなクロップ内をオプティカルフローで追跡します。検出に失敗した場合も、次のフレームでRTMPoseに戻ります。

COCO-WholeBodyの133キーポイントから、左右の足の親指を追跡します。

| オプション | キーポイント |
| --- | --- |
| `--side left` | 左足の親指（WholeBody index 17） |
| `--side right` | 右足の親指（WholeBody index 20） |

カメラを使う場合:

```powershell
.\.venv\Scripts\python.exe toe_live-RTMpose.py --side left
```

動画を使う場合:

```powershell
.\.venv\Scripts\python.exe toe_live-RTMpose.py --video path\to\input.mp4 --side left
```

CPUで実行する場合:

```powershell
.\.venv\Scripts\python.exe toe_live-RTMpose.py --video path\to\input.mp4 --cpu
```

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--rtmpose-mode` | `lightweight` | WholeBodyモデルの軽さ・精度プリセット |
| `--confidence` | `0.35` | 人物・キーポイントの信頼度しきい値 |
| `--imgsz` | `640` | 互換用オプション（RTMPoseではモデル設定を使用） |
| `--detect-every` | `30` | RTMPoseを再実行する間隔（フレーム） |
| `--crop-size` | `160` | オプティカルフロー追跡範囲の大きさ（px） |
| `--trail-length` | `90` | 画面に表示する直近軌道の点数 |
| `--device` | `cpu` | `cpu`、`cuda`、`mps`などの推論デバイス |
| `--csv` | なし | `frame,x,y,source`形式のCSV出力先 |

例:

```powershell
.\.venv\Scripts\python.exe toe_live-RTMpose.py `
  --video path\to\input.mp4 `
  --rtmpose-mode lightweight `
  --side right `
  --detect-every 10 `
  --trail-length 120 `
  --confidence 0.4 `
  --csv results\right_ankle.csv
```

画面表示中に `q` キーを押すと終了します。CSVの `source` は、RTMPoseによる検出が `pose`、オプティカルフローによる追跡が `track`、未検出が `none` です。

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

`rtmlib`のWholeBodyは、初回実行時に人物検出器とRTMW WholeBody ONNXモデルを`%USERPROFILE%\\.cache\\rtmlib`へ自動取得します。`--rtmpose-mode lightweight`はCPUで試す場合に適しています。
