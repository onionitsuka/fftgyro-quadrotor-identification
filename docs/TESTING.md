# テスト

```sh
python -m pytest -q
python -m pytest -q sim/appendix/tests
python scripts/check_publication.py
```

解析の数式・不変条件、合成信号での同定、SITL の荷重計算、通信モック、設定の検証等を対象としています。
試験条件や承認記録を使う検証は合成入力で行い、実際の試験計画・実施履歴を必要としません。

Pitch commissioning の合成 fixture に、native reference capture 条件を満たさない
既知の1ケースがあり、理由を付けた strict xfail としています。
実装の開始条件は緩めていません。

ローカル HTTP/UDP テストには loopback socket の許可が必要です。
macOS の一時ディレクトリの symlink 差を避ける場合は `TMPDIR=/private/tmp` を指定してください。

Python の通常テストは、C++ executor のビルド、PX4/Gazebo の統合動作、
実データでの再同定や実機動作の検証を含みません。
