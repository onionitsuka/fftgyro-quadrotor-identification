# FFTGYRO quadrotor identification and SITL

三軸姿勢試験台 FFT GYRO を用いたクアッドロータの荷重・推進応答の同定と、
同定モデルを使ったオフラインシミュレーション／PX4 SITL の研究コードです。

解析・同定、収録用ツール、マイコン／Raspberry Pi の実装、PX4 パッチを収録しています。
実験ログ、解析入力 bundle、第三者のソースツリーは含みません。
**データなしで実行できる例は `sim/appendix/`、論文解析の入口は `analysis.py` です。**

## はじめに

Python 3.11 以上を使い、このリポジトリのルートで実行してください。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[analysis,hardware,dev]'
.venv/bin/python -m pytest -q
```

Windows PowerShell では最初のコマンドを `py -3 -m venv .venv`、
以降の `.venv/bin/python` を `.\.venv\Scripts\python.exe` に置き換えてください。
Python の数値処理だけなら `hardware` は不要です。全テストには収録系の依存も使います。
PX4・Gazebo・C++ のビルド環境は Python パッケージに含まれません。
対象と既知の制約は [検証範囲](docs/TESTING.md) を参照してください。

## 入口と再現できる範囲

| やりたいこと | 入口 | 追加で必要なもの |
| --- | --- | --- |
| 保存済み係数で飛行例を計算する | [sim/appendix/](sim/appendix/README.md) | Python 依存のみ |
| 荷重を再構成し、モデルを同定・評価する | [analysis.py](analysis.py) | 実験入力 bundle |
| 保存済み結果から図・表を出す | [analysis_output.py](analysis_output.py) | 解析結果ディレクトリ |
| 現用の荷重モデルを PX4 と組み合わせる | [scripts/run_current_px4_sitl.py](scripts/run_current_px4_sitl.py) | 外部 PX4 ビルド、同定係数・機体設定 |
| Gazebo の機体を使う比較を行う | [scripts/run_native_gazebo_sitl.py](scripts/run_native_gazebo_sitl.py) | 外部 PX4/Gazebo、モデル成果物 |
| 収録・同期処理を読む | [src/fftgyro_quad_id/](src/fftgyro_quad_id/) | 実機運用には装置固有の構成が必要 |

付録シミュレーターは共通の簡易制御器を使います。PX4 SITL や実飛行の再現とは別の計算例です。
MoCap の記録飛行を使う過去の採用版比較は、外部の凍結実行環境とログを必要とするため収録していません。

## データなしで試す

```sh
.venv/bin/python sim/appendix/simulate.py --output sim/appendix/results/example
.venv/bin/python sim/appendix/simulate.py --output sim/appendix/results/example --plot-only
```

比較図は `sim/appendix/results/example/comparison.png` に保存されます。
既存の結果を保つため、再計算時は新しい出力先を指定してください。
ブラウザでの操縦例や Windows の手順は [付録の README](sim/appendix/README.md) を参照してください。

## 解析・同定

入力 bundle を用意した場合の例です。入力形式は [解析入力と出力](docs/ANALYSIS.md) に記載しています。
実験データは配布していないため、このコマンドだけで論文の数値結果を再現することはできません。

```sh
.venv/bin/python analysis.py --bundle /path/to/input-bundle --output output/analysis
.venv/bin/python analysis_output.py --input output/analysis --output output/report
```

`--refit` で保存済み構造に対する係数再推定、`--six-axis --refit` で追加 3 軸を含む
同定・評価を実行します。項の再選択や感度解析は明示的なオプションです。
図・表だけを変更する場合は `analysis_output.py` だけを実行してください。

## コードの構成

| パス | 内容 |
| --- | --- |
| `analysis.py`, `analysis_search.py` | 再構成、回帰、候補探索、誤差計算 |
| `analysis_output.py` | 保存済み結果から図・表を出力 |
| `src/fftgyro_quad_id/` | 同期、座標変換、同定、装置通信などの共通実装 |
| `scripts/` | 現行の起動処理とその依存23ファイル。[入口一覧](scripts/README.md) |
| `tests/` | 合成信号、数式の不変条件、通信モック等を使うテスト |
| `configs/` | 設定例とテスト・実装が参照する条件。区分は [設定の扱い](docs/CONFIGURATIONS.md) を参照 |
| `sim/appendix/` | 保存済み係数による単独シミュレーター |
| `onboard_executor/`, `firmware/`, `hardware/` | Pi 実行系、マイコンのコード、回路設計 |
| `protocol/`, `patches/px4/` | MAVLink 定義と外部 PX4 へ適用するパッチ |

既存の import・パス参照を保つため、研究用コードのフォルダ名は維持しています。
`analysis.py` から読み始め、SITL は [スクリプトの入口](scripts/README.md) を参照してください。

## 先行研究

モデルの項選択には、Sun et al.（2019）の前進・後退ステップワイズ回帰を用いています。
F検定の閾値とPSEによる停止条件は、著者らの公開MATLAB実装を参照し、Pythonで計算しています。

- [Sun, de Visser, and Chu (2019), Quadrotor Gray-Box Model Identification from High-Speed Flight Data](https://doi.org/10.2514/1.C035135)
- [著者らのMATLAB参照実装（固定コミット）](https://github.com/SihaoSun/stepwise_regression/tree/b04095ccb9f0e5dcafae43db6ec1a4320cbd0b16/func)

## 比較条件と解釈

- 実験の物理モデル比較は、実測 RPM がある同一標本で行います。欠測を推算 RPM で補いません。
- `measured` と `command_route` は学習・予測で使う RPM の経路です。推算経路の fit に実測 RPM を混ぜません。
- シミュレーション中の RPM は生成値です。実測 RPM と区別してください。
- 全学習データへの一致と条件別検証を分け、独立停止基準とモデル補助基準を分けます。
- 試験台上の荷重一致だけで自由飛行の精度を保証しません。RTS やゼロ位相フィルタはオフライン処理です。

## ライセンスと外部依存

プロジェクト全体のライセンスは未設定です。
外部依存は [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)、
第三者由来部分の出典とライセンスは [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。
