# 外部依存

Python の依存は `pyproject.toml` に定義しています。
`analysis` は NumPy、SciPy、pandas、Matplotlib、PyArrow、pyulog、
`hardware` は pyserial と pymavlink、`dev` は pytest、Ruff、mypy です。
MuJoCo を使う運動学確認は `kinematics-check` を追加します。

PX4-Autopilot、Gazebo とその Python bindings は外部に用意してください。
この公開版はそれらのソース、SITL バイナリ、Gazebo の生成モデルを同梱しません。
パッチの基準コミットと適用手順は `patches/px4/README.md` を参照してください。
パッチを使う実機収録系と、標準 PX4 の制御器を使うシミュレーションの要件を混同しないでください。

`onboard_executor/` のフルビルドには CMake、C++20、nlohmann/json、外部 MAVLink generator が必要です。
GPIO を使う Linux の構成では libgpiod も使用します。詳細は CMakeLists.txt に定義しています。
`FFTGYRO_RPM_ONLY=ON` は MAVLink を使わない RPM 受信器だけのビルドです。

Python のインストールで利用可能になるのは `src/` のパッケージです。
`analysis.py` と `scripts/` を使う研究ワークフローは、ソースの checkout から実行してください。
