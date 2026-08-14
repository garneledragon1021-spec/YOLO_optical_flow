# RTMPose BodyWithFeet + Optical Flow サンプル

`rtmlib`のRTMPose BodyWithFeetとOpenCVを使って、つま先の検出、追跡、座標CSV化、角速度計算、注釈動画作成を行うスクリプト群です。RTMPoseモデルは初回実行時に自動でダウンロードされます。

前提:
- Windows / macOS / Linux
- Python 3.11 系
- `rtmlib` / ONNX Runtime
- カメラを使う場合はOSのカメラ許可

プログラム概要:

主に実行するスクリプト:

| ファイル | 概要 |
| --- | --- |
| `toe_mp.py` | 動画の全フレームを MediaPipe Pose で処理し、左足先ランドマークの `x`, `y` 座標を `results/<動画名>_toe_mp.csv` へ保存します。`toe_flow.py` との比較用の基準データに使えます。 |
| `toe_live.py` | カメラまたは動画上で左右どちらかのつま先をリアルタイム追跡します。RTMPose BodyWithFeetは一定間隔だけ実行し、間のフレームは小さいクロップ内のオプティカルフローで追跡します。 |
| `toe_flow.py` | 動画ファイルを対象に、つま先座標をRTMPose BodyWithFeetとLucas-Kanadeオプティカルフローで追跡します。生座標とSavitzky-Golayフィルター後座標をCSVに出力し、必要なら注釈付きMP4も作成します。 |
| `trajectory_angle.py` | `frame`, `x`, `y` 系の座標 CSV から、軌跡の進行方向角と角速度を計算して新しい CSV を作成します。`--fps` を指定すると秒単位の角速度も出力します。 |
| `overlay_video.py` | フィルター後座標 CSV を元動画へ重ね、BB 中心まわりの角速度に応じて軌跡の色を変えた MP4 を書き出します。元動画・CSV・BB 中心は GUI または引数で指定できます。 |
| `foot_flow.py` | 足首・踵・つま先から作った足部 ROI 内の複数特徴点を疎なオプティカルフローで追跡し、BB 中心まわりの角速度中央値を `deg/frame`, `deg/s`, `rpm` として出力します。 |
| `marker.py` | 足元に付けた蛍光色マーカを OpenCV の HSV 色抽出で検出し、マーカ中心の `x`, `y` ピクセル座標を CSV に出力します。 |
| `pedal_velocity.py` | BB 座標とペダル座標の CSV から、ペダルの角度、角速度、rpm、BB 基準の相対速度、接線速度を計算します。 |

共通モジュール:

| ファイル | 概要 |
| --- | --- |
| `video_io.py` | OpenCVのMP4 writer作成を提供する小さな共通ヘルパーです。 |

セットアップ:

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

  Windows (PowerShell):

  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  ```

  実行時は必ず仮想環境の Python を使ってください。たとえば Windows では:

  ```powershell
  .\.venv\Scripts\python.exe toe_live.py --side left --cpu
  ```

全フレームを MediaPipe Pose で処理してCSVを出力する場合:

```bash
python3 toe_mp.py
```

引数を省略すると `tkinter` のファイル選択ダイアログを使います。
`--video /path/to/video.mov` でも指定できます。出力先は `results/<動画名>_toe_mp.csv` です。

つま先だけをリアルタイム追跡したい場合:

```bash
python3 toe_live.py --side left
```

動画ファイルで試す場合:

```bash
python3 toe_live.py --video /path/to/video.mov --side left --detect-every 30
```

`toe_live.py` はRTMPose BodyWithFeetを毎フレーム実行せず、一定間隔だけ検出して、その間はつま先周辺の小さいクロップをOpenCVのオプティカルフローで追跡します。`--detect-every`を大きくするとRTMPoseの実行回数が減り、`--crop-size`で追跡範囲を調整できます。

`toe_flow.py` で動画処理する場合:

```bash
python3 toe_flow.py --video /path/to/video.mov --side left --detect-every 30 --write-video
```

`toe_mp.py` は従来のMediaPipe版です。`toe_flow.py` はRTMPose BodyWithFeetを一定間隔だけ実行し、その間をオプティカルフローで追跡します。

出力CSVにはフィルタ前の座標 `x`, `y` と、Savitzky-Golayフィルタ後の座標
`x_savgol`, `y_savgol` が格納されます。既定値は窓幅11、多項式次数2です。
変更する場合は `--savgol-window` と `--savgol-polyorder` を指定してください。
窓幅3・多項式次数2のように、多項式次数を窓幅マイナス1にすると入力座標を
そのまま再現して平滑化されないため、指定できません。
検出できなかったフレームは空欄のまま保持し、フィルタは欠損を除いた有効座標列に適用されます。

1点の座標軌跡から、進行方向角とその角速度を計算する場合:

```bash
python3 trajectory_angle.py results/savitky-goley.csv
python3 trajectory_angle.py results/savitky-goley.csv --fps 30
python3 trajectory_angle.py results/savitky-goley.csv --omega-savgol-window 15
```
  ```powershell
  .\.venv\Scripts\python.exe trajectory_angle.py "results\your_output.csv"
  .\.venv\Scripts\python.exe trajectory_angle.py "results\your_output.csv" --fps 30
  .\.venv\Scripts\python.exe trajectory_angle.py "results\your_output.csv" --omega-savgol-window 15
  ```

  `toe_mp.py` または `toe_flow.py` が出力した CSV を `results\your_output.csv` の部分に置き換えて実行してください。日本語やスペースを含むファイル名は必ず引用符で囲んでください。角速度計算済みの CSV を再度渡すと、こちらは入力として扱えません。

角速度は軌跡の速度ベクトルが回転する速さです。画像Y軸を反転した直交座標系で、
反時計回りを正とします。足関節などの関節角速度を求めるには、つま先以外の
ランドマークも必要です。通常の数値微分角速度 `omega_*` に加え、アンラップした
進行方向角をSavitzky-Golay微分した `omega_savgol_*` を出力します。角速度用の
既定値は窓幅11・多項式次数2です。

フィルター後座標を元動画へ重ねてMP4を書き出す場合:

```bash
python3 overlay_video.py
```

引数を省略すると、元動画・CSV・表示する窓幅・出力先をGUIで選択できます。
コマンドラインで指定する場合:

```bash
python3 overlay_video.py \
  --video /path/to/IMG_2017.mov \
  --csv results/IMG_2017_toe_flow.csv \
  --bb-frame 0 \
  --angular-speed-threshold 180 \
  --show-raw \
  -o results/IMG_2017_filtered.mp4
```

`savitky-goley.csv` の窓幅11を表示する場合は `--suffix w11` を指定します。
フィルター後座標と直近の軌跡は、進行方向の絶対角速度に応じて緑から黄を経て
赤へ連続的に変わります。`--angular-speed-threshold` は完全な赤になる角速度で、
既定値は180 deg/sです。角速度を計算できない箇所はシアンになります。
`--show-raw` 指定時のフィルター前座標も
赤い十字で表示されます。
角速度は `--bb-x` と `--bb-y` で指定した自転車のBB中心からつま先へ向かう
BB中心座標系のベクトルの角度変化として計算します。x は右方向、y は上方向を正にするため、
MediaPipe/OpenCV の画像 y 座標とは符号を反転します。BB中心は白い斜め十字で表示されます。
`--bb-frame 0` のようにフレーム番号を指定すると、そのフレームを表示して
クリックした位置をBB中心にできます。クリック後にEnterキーで確定します。

足部全体の複数特徴点を疎なオプティカルフローで追跡し、BB周り角速度の中央値を
推定する場合:

```bash
python3 foot_flow.py \
  --video /path/to/IMG_2017.mov \
  --side left \
  --bb-frame 0
```

指定フレーム上でBB中心をクリックしてEnterキーで確定します。MediaPipeで
足首・踵・つま先から足部ROIを定期更新し、ROI内のShi-Tomasi特徴点を
人物セグメンテーションマスクとの交差領域から抽出してLucas-Kanade法で追跡します。
これにより足部の背後にある機材上の特徴点を除外します。前後追跡誤差と角速度のMADで外れ値を除外し、
静止背景に近い点やBB周り角速度がほぼゼロの機材点も除外して、残った点の角速度中央値を
`deg/frame`、`deg/s`、`rpm` としてCSVへ出力します。
注釈動画では追跡点が角速度に応じて緑から赤へ変化します。

蛍光色マーカの座標を OpenCV の色抽出で取得する場合:

```bash
python3 marker.py --video /path/to/IMG_2017.mov --bb-frame 1000
python3 marker.py --video /path/to/IMG_2017.mov --bb-x 1075 --bb-y 919 --show-mask
```

出力先は既定で `results/<動画名>_marker.csv` と、検出位置を重ねた
`results/<動画名>_marker.mp4` です。動画が不要な場合は `--no-write-video` を指定します。
既定では BB 部分の赤
マーカと、シューズ部分の黄色マーカを同時に検出し、`bb_x`, `bb_y`, `shoe_x`,
`shoe_y` に画像左上を原点にしたマーカ中心のピクセル座標を出力します。
`--bb-frame` には赤マーカが見えるフレームを指定し、表示された画像上でBB中心を
クリックしてEnterキーで確定します。座標が分かっている場合は `--bb-x` と `--bb-y` を
指定できます。BBは選択位置の近傍だけを探索し、他の赤い物体への誤移動を防ぎます。
赤マーカがクランクや足で隠れたフレームでは直前の信頼できるBB座標を使用し、CSVの
`bb_detected=0`, `bb_estimated=1` で直接検出値と区別します。BB指定を省略した場合は
先頭フレームでクリック選択画面が開きます。
蛍光色がプリセットに合わない場合は、1色検出モードで OpenCV の HSV 範囲を調整できます。

```bash
python3 marker.py --video /path/to/IMG_2017.mov --color yellow --lower-hsv 20,40,50 --upper-hsv 60,255,255
```

BB 座標とペダル座標から回転速度・移動速度を計算する場合:

```bash
python3 pedal_velocity.py
python3 pedal_velocity.py results/IMG_2017_marker.csv --fps 30
```

CSV パスを省略すると `tkinter` のファイル選択ダイアログを使います。引数なしで
起動した場合は、CSV 選択後に角速度色付きマーカを重ねる元動画と出力 MP4 も GUI で
選択できます。元動画の選択をキャンセルすると CSV 出力だけを行います。

足元マーカ位置を動画上に常時表示し、角速度が速い箇所を色で確認したい場合は、
引数で元動画を渡すこともできます。

```bash
python3 pedal_velocity.py results/IMG_2017_marker.csv --fps 30 \
  --video /path/to/IMG_2017.mov
```

出力動画は既定で `/path/to/IMG_2017_pedal_velocity_marker.mp4` です。ペダル側の
座標位置に常時ポイントを表示し、`|omega|` が小さい箇所は緑、速い箇所は赤へ
変化します。赤になる閾値は未指定なら1回転ごとの角速度ピークから自動調整されるため、
スローモーション動画でも周期ごとの速い箇所を比較しやすくなります。固定したい場合は
`--angular-speed-threshold` で指定できます。

`marker.py` の既定出力では `bb_x`, `bb_y` を BB、`shoe_x`, `shoe_y` をペダル側の
座標として読みます。別の列名を使う場合は `--bb-prefix` と `--pedal-prefix`、
または `--bb-x-column` などで指定できます。出力には、BB とペダルそれぞれの
画像上の速度、BB 基準の相対速度、BB を原点にした `bb_centered_x`, `bb_centered_y`、
BB からペダルへ向かうクランク角と角速度
`omega_*`、ケイデンス相当の `rpm`、半径方向速度と接線速度が入ります。
`bb_centered_x = pedal_x - bb_x`, `bb_centered_y = pedal_y - bb_y` で、MediaPipe/OpenCV の
画像座標のまま x は右方向、y は下方向を正にします。`angle_deg` は
`atan2(bb_centered_y, bb_centered_x)` を 0..360 度に正規化したクランク角です。

`toe_flow.py` の MediaPipe Tasks CPU 初期化だけ確認したい場合:

```bash
python3 toe_flow.py --check
```

補足:
- 初回は `pose_landmarker_full.task` を `~/.cache/mediapipe/` に保存します。
- `--cpu` は古い実行コマンドとの互換用です。現在は指定しなくても常に CPU delegate を使います。
- Apple Silicon では `delegate=CPU (Apple Silicon)` と表示されます。
- カメラの取り込みと OpenCV の表示はこの最小構成では CPU です。
- MediaPipe Pose はつま先だけの専用推論にはできないため、クロップは検出後の追跡負荷を下げる目的で使います。
- この venv は Tk 対応の Python で作り直してあり、`tkinter` が使えます。
