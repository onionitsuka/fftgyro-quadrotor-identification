# 設定例

- `configs/experiment_runner/*.example.*`：収録装置・電源・入力信号等の設定例。
- `configs/experiment_runner/empty-sensor-static-qc.json`：静止収録の品質検査の閾値。
- `configs/dynamic_wrench_pitch.template.json`：荷重試験のテンプレート。
- `configs/analysis/freeflight-coverage.example.json`：飛行条件の記述例。
- `onboard_executor/config/*.example.json`：Pi 実行系の設定例。

実際の装置に合わせる変更は、Git 管理外の `configs/local/` に置いてください。
`192.0.2.*` は例示用のアドレス、`/home/fftgyro/` はサービス用のパスです。
設定例の数値や配線・軸対応・停止条件は、使用する装置に合わせて確認してください。

個別の試験計画、実施履歴、実機の承認記録・校正記録は配布していません。
ライブラリの一部にある研究時のカタログ処理を使う場合は、呼び出し側で入力を用意してください。
