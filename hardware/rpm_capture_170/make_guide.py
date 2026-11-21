# ruff: noqa: E501
# Long strings below are generated CAD/HTML text, not control logic.
"""Generate the hole table and a self-contained, selectable assembly drawing."""

import html
import json
from pathlib import Path

import layout_work as m

root = Path(__file__).resolve().parent
d = json.loads((root / "layout.json").read_text())
svg = [
    '<svg id="board" viewBox="-4 -2 20 22" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="170穴の配置図">'
]


def add(s):
    svg.append(s)


def txt(x, y, s, size=0.27, fill="#304155", anchor="middle"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">{html.escape(str(s))}</text>'


add(
    '<rect x="-.65" y="-.65" width="12.3" height="17.3" rx=".2" fill="#fff" stroke="#a4afbd" stroke-width=".05"/><rect x="4.6" y="-.65" width="1.8" height="17.3" fill="#e9edf3"/>'
)
for r in range(1, 18):
    for x1, x2 in [(0, 4), (7, 11)]:
        add(f'<path d="M{x1} {r - 1}H{x2}" stroke="#e0e5ec" stroke-width=".12"/>')
    add(txt(-3.2, r - 1, r, 0.3))
    for c, x in m.COLS.items():
        add(f'<circle cx="{x}" cy="{r - 1}" r=".08" fill="#aeb8c5"><title>{c}{r}</title></circle>')
for c, x in m.COLS.items():
    add(txt(x, -1.15, c, 0.36))
add(
    '<g id="chip"><rect x="4.25" y="4.6" width="2.5" height="6.8" fill="#26364d" rx=".1"/><circle cx="5.5" cy="4.6" r=".35" fill="#e9edf3"/>'
)
add(
    '<text x="5.5" y="8.9" font-size=".32" fill="white" text-anchor="middle" transform="rotate(-90 5.5 8.9)">TC74HC14AP</text>'
)
for n, h in d["ic"]:
    x, y = m.xy(h)
    end = 4.25 if h[0] == "E" else 6.75
    add(f'<path d="M{x} {y}H{end}" stroke="#747f8f" stroke-width=".08"/>')
    add(txt(x + (0.3 if h[0] == "E" else -0.3), y - 0.16, n, 0.2, "#f6c578"))
add("</g>")
col = {"GND": "#506479", "SENSOR_5V": "#df424c", "USB_5V": "#d8920c"}
for i, w in enumerate(d["wires"], 1):
    pts = " ".join(f"{x},{y}" for x, y in w["path"])
    add(
        f'<g class="item wire" id="W{i}"><title>W{i}: {w["a"]} – {w["b"]} / {w["net"]}</title><polyline points="{pts}" fill="none" stroke="{col[w["net"]]}" stroke-width=".065" stroke-linejoin="round"/>'
    )
    for h in [w["a"], w["b"]]:
        x, y = m.xy(h)
        add(
            f'<circle cx="{x}" cy="{y}" r=".085" fill="white" stroke="{col[w["net"]]}" stroke-width=".045"/>'
        )
    add("</g>")
for p in d["parts"]:
    a, b = m.xy(p["a"]), m.xy(p["b"])
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if p["ref"] == "RO1":
        cx, cy = a[0], a[1] - 0.4
    add(
        f'<g class="item part" id="{p["ref"]}"><title>{p["ref"]} {p["value"]}: {p["a"]} – {p["b"]}</title><path d="M{a[0]} {a[1]}L{b[0]} {b[1]}" stroke="#3d4b5d" stroke-width=".07"/>'
    )
    if p["ref"].startswith("R"):
        add(
            f'<rect x="{cx - 0.27}" y="{cy - 0.5}" width=".54" height="1" fill="#e3bc84" stroke="#94704c" stroke-width=".04"/>'
        )
    else:
        add(
            f'<circle cx="{cx}" cy="{cy}" r=".28" fill="{"#69a65d" if p["ref"] == "F1" else "#46a7ba"}" stroke="#345466" stroke-width=".035"/>'
        )
    add(txt(cx + 0.32, cy, p["ref"], 0.24, anchor="start"))
    add("</g>")
for i, (label, h) in enumerate(d["external"]):
    x, y = m.xy(h)
    add(
        f'<g class="item terminal" id="T{i}"><title>{html.escape(label)} → {h}</title><rect x="{x - 0.1}" y="{y - 0.1}" width=".2" height=".2" fill="white" stroke="#845ba6" stroke-width=".055"/></g>'
    )
add("</svg>")
buttons = []
for p in d["parts"]:
    buttons.append((p["ref"], f"{p['ref']}　{p['value']}　{p['a']} ↔ {p['b']}"))
for i, w in enumerate(d["wires"], 1):
    buttons.append(
        (
            f"W{i}",
            f"W{i}　{w['net']}　{w['a']} ↔ {w['b']}" + ("【ICの上】" if w.get("over_ic") else ""),
        )
    )
for i, (label, h) in enumerate(d["external"]):
    buttons.append((f"T{i}", f"{label} → {h}"))
page = (
    """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RPM回路・170穴の組立図</title><style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;color:#203149;background:#f3f5f9;margin:0;padding:20px}h1{font-size:24px;margin:0 0 8px}p{line-height:1.7;margin:6px 0 12px}main{display:grid;grid-template-columns:minmax(400px,1fr) 390px;gap:20px;max-width:1250px;margin:auto}.drawing{position:sticky;top:10px;align-self:start;background:white;border-radius:14px}svg{width:100%;max-height:86vh}button{font:inherit;cursor:pointer;background:white;border:1px solid #c3cdd9;border-radius:7px;padding:9px;text-align:left}nav{display:grid;gap:6px}button.active{background:#dbeafa;border-color:#4484b9}header{max-width:1250px;margin:auto}.tools{display:flex;gap:8px;margin:12px 0}.item{transition:opacity .1s}.muted{opacity:.12}.selected path,.selected polyline{stroke-width:.14}.selected circle{stroke-width:.07}.legend{font-size:13px;padding:10px}#current{background:#e6edf6;padding:10px;border-radius:8px;min-height:48px}@media(max-width:800px){main{display:block}.drawing{position:relative}svg{max-height:70vh}nav{margin-top:16px}}
</style><header><h1>RPM回路・170穴の組立図</h1><p>部品や線を選ぶと、その1本だけを強調します。ICの切欠きは上。Nano Everyは外置きです。<br><b>W1だけはICの樹脂ケースの上を通す被覆線です。</b> 折れ曲がりは線の通り道で、途中の穴へ差し込む場所ではありません。</p><p>線の中心同士の交差は0。部品本体の絵は概形で、購入品の寸法・高さ・外部ケーブルの取り回しは実物で確認します。</p><div class="tools"><button onclick="selectItem('')">すべて表示</button><button onclick="onlyParts()">部品だけ表示</button></div><p id="current">右の一覧から、差し込む部品・線を選んでください。</p></header><main><section class="drawing">"""
    + "".join(svg)
    + """<p class="legend">赤：センサー5V　黄：Nano 5V　灰：GND　紫□：外部への接続穴<br>W1以外はICの下へ線を押し込まず、描いた経路に沿わせます。</p></section><nav>"""
    + "".join(
        f'<button data-id="{ref}" onclick="selectItem(this.dataset.id,this.textContent)">{html.escape(label)}</button>'
        for ref, label in buttons
    )
    + """</nav></main><script>
function selectItem(id,label){document.querySelectorAll('.item').forEach(e=>{e.classList.toggle('muted',!!id&&e.id!==id);e.classList.toggle('selected',e.id===id)});document.querySelectorAll('[data-id]').forEach(e=>e.classList.toggle('active',e.dataset.id===id));document.getElementById('current').textContent=label||'全体を表示しています。'}
function onlyParts(){document.querySelectorAll('.item').forEach(e=>{e.classList.toggle('muted',!e.classList.contains('part'));e.classList.remove('selected')});document.querySelectorAll('[data-id]').forEach(e=>e.classList.remove('active'));document.getElementById('current').textContent='抵抗・コンデンサ・PTCだけを表示しています。';}
</script></html>"""
)
(root / "組立図.html").write_text(page)
# Hole table, generated from the same validated physical data.
text = """# RPM回路：170穴ブレッドボードの配線表（交差を減らした改訂版）

2026-09-07。**旧配置から全面変更。旧表へ部分的に継ぎ足さず、この表で組み直す。** Nano Everyは外置き、170穴1枚を使用。部品値とNanoのD2〜D5は維持し、74HC14の使用ゲートを左右2系統ずつへ変えた。電源の10 µFは本人購入済みの**25 V・積層セラミック、無極性**。

**先に[1本ずつ選べる組立図](../../hardware/rpm_capture_170/組立図.html)を開く。** [全体図](../../hardware/rpm_capture_170/placement.png)も参照できる。表は差し込む両端、図はその間の通し方を示す。途中の曲がり角に接続点はない。

## 1. 向きと配置

横A〜J、上から1〜17。同じ段のA〜EとF〜Jがそれぞれ内部で接続され、中央の溝をまたいだ左右は別。ICの切欠きを上にして、E6〜E12／F6〜F12へ置く。USB・電池を外して組み、1穴に2本を差さない。

| 系統 | 場所 | IC入力→出力 | Nano |
|---|---|---|---|
| RPM1 | 左上 | 1→2 | D2 |
| RPM2 | 左下 | **5→6** | D3 |
| RPM3 | 右上 | **13→12** | D4 |
| RPM4 | 右下 | 9→8 | D5 |

**未使用入力は3・11番をGNDへ。未使用出力は4・10番を未接続。** 旧表の「11・13番をGND」は使わない。14番は5V、7番はGND。ピン番号は[東芝資料](https://toshiba.semicon-storage.com/info/docget.jsp?did=10034&prodName=TC74HC14AP)と照合済み。

## 2. 部品を差す

まず部品だけ仮置きする。抵抗本体は図の位置へ寄せ、隣のコンデンサと高さ・位置をずらす。特にRO1の本体は7段寄り。10 µFと0.1 µFを隣接するG列・H列に詰める旧案をやめ、電源コンデンサ同士の間隔を空けた。

| 図の記号 | 部品 | 穴1 | 穴2 |
|---|---|---|---|
"""
for p in d["parts"]:
    text += f"| {p['ref']} | {p['value']} | {p['a']} | {p['b']} |\n"
text += """
`4.7k`＝4.7 kΩ、`1k`＝1 kΩ、`100`＝100 Ω、`1n`＝1 nF、`100n`＝0.1 µF。今回の表のコンデンサはすべて無極性。F1は抵抗ではなくMF-R050のPTC。短い穴間隔に合うよう足を曲げるが、部品本体の根元を無理に曲げない。部品本体寸法が図と同じとは限らない。

## 3. 被覆付きジャンパーを差す

**W1はD8→G9をICの樹脂ケースの上に渡す短いGND線。** ピン列に触れない被覆線を使う。IC直近のC1のGNDを左側へ短く戻すためで、ICの下に線を挟まない。ほかの線は図の経路に沿わせる。裸線の斜め横断や、両端だけ合わせた直線配線へ変更すると交差0の条件を失う。

| 線 | 接続 | 穴1 | 穴2 | 通し方 |
|---|---|---|---|---|
"""
for i, w in enumerate(d["wires"], 1):
    text += (
        f"| W{i} | {w['net']} | {w['a']} | {w['b']} | "
        + ("ICの上。図の短い橋渡し" if w.get("over_ic") else "組立図でW" + str(i) + "を選択")
        + " |\n"
    )
text += """
Nanoからの5Vと、PTCを通ったセンサー5Vは別の配線。PTC前後は通常時に導通するが、PTCをバイパスする線を足さない。図で板の外縁に回した線も、機体側の絶縁台座上に収めて固定する。

## 4. Nano Everyとセンサーをつなぐ

外部線は紫の四角の穴へ差す。Nanoは5VとGNDを含む6本。5VをVIN・3.3Vと取り違えない。NanoはPiのUSBから給電し、別の5V電源線を並列に追加しない。

| 接続元 | 穴 |
|---|---|
"""
for label, h in d["external"]:
    text += f"| {label} | {h} |\n"
text += """
センサーはEE-SPY401＋EE-1003。`+`は電源、`OUT`は信号、2つの`GND`は同じセンサーの−端子へ戻る。コネクター配線側で分岐してOUT＋GNDと電源＋GNDの2対をねじる。線の色だけで極性を決めない。

**センサー直近の0.1 µF＋1 µFは各センサー側に残す。** ここへ集めない。詳細は[部品と配線](機体搭載RPM回路_部品と配線.md)を参照。センサー端子へ直接はんだ付けしない。

外部ケーブルは図の穴から板の外へ引き出し、基板上を横断して束ねない。Nanoまでの6本とセンサーケーブルの立体的な交差は、今回の交差数には含めていない。実機への固定位置に合わせて処理する。

## 5. どこまで確認したか

- KiCad 10.0.5で回路図を読み込み、ERCは**エラー0・警告0**。
- KiCad出力ネットリストと、170穴の内部接続＋ジャンパーを別計算した接続関係が一致。
- 102穴の重複なし。200通りの接続順と200通りの再配線順を比較。**指定経路における部品リード同士・ジャンパー対リード・ジャンパー同士の中心線の交差／接触は各0**。これは投影線分の検査であり、全配置の総当たりや部品本体の3D干渉検査ではない。W1のICケース上の橋渡しは意図した立体配置。
- KiCad付属ngspiceの簡略モデルで、受信回路に2 kHz・5 msの入力と100 nsの短パルスを与え、4系統とも10パルスを出力。**実センサーの2 kHz動作を保証する結果ではない。** 74HC14は代表閾値の理想スイッチ、センサーも理想化しており、ESC由来の結合ノイズや実部品の公差・容量低下は未検証。

[検査結果](../../hardware/rpm_capture_170/verification.json)／[KiCadプロジェクト](../../hardware/rpm_capture_170/rpm_capture.kicad_pro)／[再実行方法](../../hardware/rpm_capture_170/README.md)

**通電前は実物の導通・IC向き・部品本体の干渉を確認する。** その後ESC停止状態で給電し、各センサー端の電圧と4chの遮光応答を確認する。実機の最大RPM・ノイズ・同期・振動時の接触安定性は別の成立確認として残る。
"""
(root.parents[1] / "docs/experiment_runner/RPM回路_170穴ブレッドボード配線表.md").write_text(text)
print("HTML and hole guide generated")
