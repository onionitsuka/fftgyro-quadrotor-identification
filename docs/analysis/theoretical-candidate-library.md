# 理論候補項

抗力・風向依存・局所流入・減衰を明示した候補と、その選択方法を説明します。

## 実行と選択対象

`analysis.py --bundle INPUT_BUNDLE --output NEW_OUTPUT --refit --reselect-physics`

追加3軸も含める場合は `--six-axis` を付ける。入力 bundle と計算環境を別途用意する。
運動補正停止基準を使うMy/Mz/Fzでは、既存と同様に `--zero-recovery SUMMARY --drift-prior PRIOR` を付けられる。
停止基準拡張と `--six-axis` の同時実行は従来どおり未対応である。

- My/Mz/Fz：各モデルの既存採用項と新しい理論項の和集合から、各学習fold内でSun公開版F/PSE選択を実行する。
  ridgeの強さ、損失、学習run集合は従来設定のまま。Yaw局所項だけに掛けていた追加罰則は1にし、列正規化後の罰則を共通化する。
  全ての高次多項式を再探索する機能ではない。
- Fx/Fy/Mx：既存2次／3次辞書それぞれへ理論候補を追加し、従来の内側3条件foldで辞書・正則化を選択する。
- 基礎モデルと差分モデルの学習対象は変更しない。零点不明runの荷重平均を絶対値教師にしない。
- モデル補助のYaw基準移送は従来どおり新モデルから再計算される。独立停止基準とは区別して出力し、その基準の変化を独立な精度改善と解釈しない。
- `models.json` の `selection.candidate_names` と `feature_names` に、候補と採用項を別々に保存する。
- 元の `--refit` だけなら従来どおり係数更新のみ。新オプションなしの数値処理は維持する。

全候補を必ず選択肢に入れるが、採用を強制しない。相関の強い候補を同時に採用しない場合がある。
`theory__` は出典を追跡する名前であり、優先順位や選択ボーナスには使わない。
切片がある場合の固定採用・ridge免除、および既存推進補正の係数範囲は従来どおり。
同じ波形になる候補同士は識別できず、同点なら辞書の順序に依存しうる。その採用名を物理機構の証明としない。
選択後に残らなくても、その物理作用が存在しない証拠にはならない。
My/Mz/Fzの新しい正則化探索は行わず、過去に探索済みのデータであるため、新規盲検検証とは呼ばない。

## 座標・速度・面積

機体重心のFRD座標を使い、空気に対する機体の速度を
\(\mathbf V=(V_x,V_y,V_z)\)、\(U=\|\mathbf V\|\)、機体角速度を \(\boldsymbol\omega=(p,q,r)\) とする。
空気が機体へ向かって流れる向きのベクトルとは符号が逆である。

基準面積は \(S_0=4\pi R^2\)、代表長さは既存の腕長 \(\ell\) とする。
\(S_0\) は係数の正規化に使う面積であり、測定した機体投影面積ではない。
抗力係数単独よりも \(S_0 C_D\) の積が物理的な有効抗力面積に対応する。
各候補は力ならN、モーメントならN·mの次元を持ち、係数は無次元。

## 機体抗力と風に対する姿勢

基本候補は

\[
\mathbf f_D=-\frac12\rho S_0 U\mathbf V.
\]

Fx/Fy/Fzの各成分へこの候補を入れる。各軸は独立fitのため、3方向に同一の \(C_D\) を強制するモデルではない。

係数の姿勢依存は、地面基準のEuler角を掛けず、風向の方向余弦を使う。

\[
\mathbf n=\frac{\mathbf V}{U}
=(\cos\alpha\cos\beta,\;\sin\beta,\;\sin\alpha\cos\beta).
\]

ここで \(\beta=\operatorname{atan2}(V_y,\sqrt{V_x^2+V_z^2})\) は通常の3次元横滑り角であり、
従来のSun領域分けで使う水平面の折り返し角とは定義を区別する。新候補では角度そのものを計算しない。

\[
\mathcal B=\{1,n_x,n_y,n_z,n_xn_y,n_xn_z,n_yn_z,n_y^2,n_z^2\},
\qquad f_{D,j} B_k,\quad B_k\in\mathcal B.
\]

例えば \(f_{D,x}n_z\) は \(UV_x\sin\alpha\cos\beta\) 型の項。
\(n_xn_z=\tfrac12\sin(2\alpha)\cos^2\beta\) も候補に含まれる。
\(n_x^2=1-n_y^2-n_z^2\) のため、重複する二次項は追加しない。
前後・左右・上下の違いを保持し、Yawの±180°で飛ばない。
\(U=0\) では方向余弦を便宜的に0とするが、動圧付きの荷重候補自体が0へ連続に近づく。

風向に直交する法線力も、\(j=y,z\) について

\[
\mathbf f_{N,j}=\frac12\rho S_0 U^2 n_j
\left(\mathbf e_j-n_j\mathbf n\right)
\]

を候補に入れる。これは作用を確定した式ではなく、迎角・横滑りに対する低次の法線力候補である。

## 機体モーメントと減衰

動圧 \(q_\infty=\tfrac12\rho U^2\) に対する

\[
q_\infty S_0\ell B_k,
\qquad
\bigl[(\ell\mathbf e_k)\times\mathbf f_D\bigr]_j B_l
\]

を入れる。後者は抗力作用点と重心のずれが作るモーメントの形を表す。
係数は抗力係数と無次元作用点距離の積を含むため、これだけから重心や作用点位置を独立に測定したとは言えない。
各成分を独立fitする現段階では、全6軸を共通作用点で拘束するモデルでもない。

各軸の角速度 \(\omega_j\) に対して、減衰候補は

\[
-\frac12\rho S_0\ell^2 U\omega_j,
\quad
-\rho A R\ell^2\left(\sum_i|\Omega_i|\right)\omega_j,
\quad
-\frac12\rho S_0\ell^3\omega_j|\omega_j|,
\qquad A=\pi R^2.
\]

正の係数なら回転を抑える符号。ただし今回は補正モデルの項選択であり、係数の非負やモデル全体の受動性を強制していない。
学習後の減衰符号・外挿挙動は別途確認が必要である。

## 各ローターの局所流入

\[
\mathbf V_i=\mathbf V+\boldsymbol\omega\times\mathbf r_i,
\qquad \mathbf V_{h,i}=(V_{x,i},V_{y,i},0).
\]

次の局所力候補をローターごとに計算して足し合わせる。

\[
\begin{aligned}
\mathbf f_{h1,i}&=-\rho A R|\Omega_i|\mathbf V_{h,i},\\
\mathbf f_{h2,i}&=-\rho A\|\mathbf V_{h,i}\|\mathbf V_{h,i},\\
\mathbf f_{z1,i}&=-\rho A R|\Omega_i|V_{z,i}\mathbf e_z,\\
\mathbf f_{z2,i}&=-\rho A\|\mathbf V_{h,i}\|^2\mathbf e_z,\\
\mathbf f_{z3,i}&=-\rho A V_{z,i}|V_{z,i}|\mathbf e_z.
\end{aligned}
\]

対応するモーメントは \(\sum_i\mathbf r_i\times\mathbf f_i\) で作る。
さらにローター別の \(-\rho A R^2\Omega_i^2\mathbf e_z\) と、その腕によるモーメント、
Yawには回転方向符号 \(s_i\) 付きの \(s_i\rho A R^3\Omega_i^2\) を候補にする。
これらは既存推進モデルへの補正であり、既存推力を二重に固定加算するものではない。
誘導流を含む従来のFz採用項も候補に残す。ここで完全な後流・ローター間干渉モデルを解いたわけではない。

ローター位置には現在の幾何・重心を使う。未測定のローター面高さを追加で仮定しない。
実測RPM経路と指令＋電圧RPM経路の候補は、それぞれのRPMから独立に作る。
比較用物理モデルは常に実測RPM。欠測RPMの推算値への置換はしない。
基底関数がローター停止時に有限でも、既存の有効サンプルマスクは変更しない。

## 出典と今回の拡張

- Sun, de Visser, Chu (2019), *Quadrotor Gray-Box Model Identification from High-Speed Flight Data*,
  [DOI:10.2514/1.C035135](https://doi.org/10.2514/1.C035135)。式25のローター面内力・機体速度二乗、式32の推力流入補正、式37–43の速度・角速度・差動による候補構築を参照。
- Faessler, Franchi, Scaramuzza (2018), *Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories*,
  [本文・式2–4](https://arxiv.org/html/1712.02402v3#S3)。線形速度抵抗と角速度依存モーメントを扱う。今回の候補全体や係数をこの論文から移植したわけではない。
- 項選択は [Sun公開コード](https://github.com/SihaoSun/stepwise_regression) に由来する既存のF/PSE規則を再利用。

\(U\mathbf V\) の3次元ベクトル抗力、方向余弦による低次係数展開、局所力の6軸への展開は今回の設計である。
加速度・角加速度を経験的な空力候補へ追加していない。既に荷重再構成で使う慣性項を再びfitして誤差を吸収することを避ける。
重力、センサ自己荷重、零点、時間同期は変更しない。
