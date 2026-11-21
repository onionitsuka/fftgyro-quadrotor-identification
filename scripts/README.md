# スクリプトの入口

論文の解析はリポジトリ直下の `analysis.py`、図表は `analysis_output.py` から実行します。
このディレクトリには、その解析と現行シミュレーション／収録に必要な23個の Python ファイルを残しています。

## オフラインシミュレーション

| 入口 | 用途 |
| --- | --- |
| `run_current_px4_sitl.py` | 現行の固定基準荷重モデルと実際の PX4 制御器を組み合わせる |
| `run_native_gazebo_sitl.py` | 外部 PX4/Gazebo の機体モデルを使う |
| `run_matched_px4_sitl.py` | 同じ PX4 とミッションによる比較の共通起動処理 |
| `simulate_current_graybox.py` | 現行の絶対値＋差分モデルによる Python 単独計算 |
| `simulate_graybox_forward_flight.py` | Python の剛体運動・共通制御・推進荷重と基本計算 |
| `export_native_gazebo_trace.py` | 保存済み Gazebo ログから時系列を出力する |

実行には各スクリプトの `--help` と必要なモデル成果物を確認してください。
データ・係数 bundle がなくても動く計算例は `../sim/appendix/` です。

## 解析と評価の内部実装

`compare_candidate_dictionary.py`、`joint_force_fit.py`、`joint_force_search.py`、
`refit_theory_forces.py`、`six_axis_theory_bank.py` は候補評価・同定の共有処理です。
`recover_motion_zero_anchors.py` は独立停止基準の回収処理です。

`current_graybox_aircraft.py`、`matched_px4_plant.py`、`speed_scheduled_graybox.py`、
`build_native_gazebo_model.py`、`gazebo_identified_system.py` はモデル・PX4/Gazebo の接続部です。
`simulate_koil_pass.py`、`run_koil_px4_sitl.py`、`hover_pitch_profile.py` は
共通起動処理から参照する経路・姿勢指令等の実装です。

## 収録と公開前検査

`run_sync_bench.py` と `capture_frame_encoder.py` は実験収録の実装です。
設定例だけで当時の実験環境が復元されるものではありません。
`check_publication.py` は Git 管理対象へのデータ・認証情報・個人環境の混入をパターン検査します。

