# RPM capture board

KiCad の回路・基板設計と、配置・経路・組立図を生成する補助コードです。
`rpm_capture.kicad_pro` を KiCad で開いてください。

- 設計ソース：`rpm_capture.kicad_sch`、`rpm_capture.kicad_pcb`、`RPM.kicad_sym`。
- 配置情報：`layout.json`。`previous_layout.json` は比較検証に使う前の配置です。
- 生成・検査：`layout_work.py`、`refine_routes.py`、`make_schematic.py`、`verify_and_render.py`、`make_guide.py`。
- SPICE 計算：`receiver_demo.cir`、`run_simulation.py`。
- 図と検査記録：SVG、組立図 HTML、`verification.json`、`erc.rpt`。

生成スクリプトはこのフォルダ内の設計・出力を更新するため、試す場合はコピーを使ってください。
Python の通常テストには含めていません。KiCad／SPICE と描画の実行環境は別途必要です。
