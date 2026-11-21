# 解析入力と出力

## 入力 bundle

`analysis.py` は `--bundle` のディレクトリに以下を要求します。

```text
input-bundle/
  profile.json
  zero_covariance.npz
  signals/
    <run>.npz
```

`profile.json` の schema は `fftgyro-final-20261013-v1` です。
設定、センサ校正、モーターマップ、run 一覧、独立停止基準、採用済みモデル、Yaw の基準移送情報を
保持します。`signals/<run>.npz` は同期済みの入力信号です。入力は予測荷重を教師値として作るものではありません。
`zero_covariance.npz` は零点校正の共分散です。厳密なフィールドと配列の解釈は
`analysis.py` の `prepare_inputs()` と `main()` を参照してください。

これらは実験データなので公開版に含めていません。現在、一般データ向けの
インポーターや公開サンプル bundle はありません。実データを持たずに数式を確認する場合は
`tests/test_final_analysis.py`、`tests/test_analysis_search.py`、
`tests/test_analysis_output.py` の合成入力を参照してください。


## 実行モード

```sh
# 保存済みモデルで再構成・評価
python analysis.py --bundle /path/to/input-bundle --output output/evaluation

# 採用構造を固定した係数の再推定
python analysis.py --bundle /path/to/input-bundle --output output/refit --refit

# Fx・Fy・Mx を追加した同定・評価
python analysis.py --bundle /path/to/input-bundle --output output/six-axis --six-axis --refit

# 図表だけを出力（再同定しない）
python analysis_output.py --input output/refit --output output/report
```

`--reselect-physics` や `--systematic-search` は候補選択を変える研究用オプションです。
既存結果の再出力と、新たなモデル探索を区別してください。
`--raw-root` と `--aligned-root` は生ログを再読込して入力信号を照合する場合に使用します。

## 出力

解析は `models.json`、`summary.json`、`predictions.npz`、`yaw-transfer.json`、
`reconstructed/` 等を出力します。図表側は保存済みの結果を読み、
`metrics.csv`、`per_run.csv`、比較図、入力ファイルのハッシュを持つ manifest 等を出力します。
CSV の荷重単位は N／N·m、モーメント図の表示は mN·m です。

既存ディレクトリへの上書きを前提にしません。入力・解析出力・図表出力には別のパスを指定してください。
生成されたデータと図表は Git 管理の対象外です。
