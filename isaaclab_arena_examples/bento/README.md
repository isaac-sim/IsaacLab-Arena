# Yaskawa NHC12 弁当盛り付けデモ

MOTOMAN NEXT NHC12（6軸）と幅広の開閉トングで、ご飯・卵焼き・唐揚げ・ブロッコリーを
仕切り付き弁当箱の4区画へ順に移す、独立したIsaac Labのサンプルです。
学習済みポリシーは不要です。Arenaの環境レジストリには登録していません。

## 今回の改善と現在の状態

弁当箱を丸みのある黒い容器と赤い内側に変更し、仕切りを低くしました。
ご飯は米粒・梅干し・胡麻、卵焼きは3切れと巻いた層、唐揚げは3個と衣、
ブロッコリーは茎・房・細かいつぼみを持つメッシュに変更しています。
食材の供給位置には金属製の小皿を置きました。頂点色とメッシュからの影を描画します。

ロボットの面ごとの材質指定（`GeomSubset`）も読み取るようにしました。
公式USDの青いYASKAWAロゴ、青い土台、緑のリングを描画します。
材質の境界で頂点を分け、白い外装とロゴの色が混ざることを防いでいます。

吸着カップを取り除き、開閉するパドル型トングへ変更しました。
ご飯はまとまった一食分、他の食品も複数個をまとめた剛体として扱います。
トングの開閉は形状のアニメーションで、保持は理想的な固定拘束です。
顎の接触・摩擦・握る力や、米粒をすくう動作を物理的に検証したものではありません。

**改善版の全工程を実行し、全4品の解放・区画内配置・静止判定が成功しました。**
最終版は16,606ステップ、物理時間138.38秒で終了コード0でした。
初期の2回は、大きくしたご飯の左端が箱の左縁に掛かり、配置がタイムアウトしました。
配置時の速度を0.05 m/sへ下げ、指令と実測TCPの距離が6 mm以上のときは
下降指令の進行を止めて解消しました。移送・配置の到達誤差は3 mm以内にしています。
トングの開放幅も食品ごとの幅＋20 mmにし、箱内で開く余裕を増やしました。

最終版の成果物はコンテナ内ローカルの次の場所です。

- 最新の保存先: `/tmp/arena-runtime/outputs/bento-logo/`
- 実行結果: `result.json`
- 実測姿勢と形状: `scene.npz`（10 fps、1,385時点）
- 全体の動画: `bento.mp4`
- 食材を見やすい近接動画: `bento-closeup.mp4`
- 盛り付け後の画像: `plated-bento.png`
- ロゴの近接画像: `preview-robot-logo.png`
- 全体の連番画像: `frames.zip`
- 近接の連番画像: `frames-bento.zip`
- 記録したトングと箱の隙間の確認: `jaw-clearance-review.json`
- 動画・連番画像の検証結果: `media-verification.json`
- 形状・法線・動作の一致確認: `material-refresh.json`

ロゴ対応版は、`bento-realistic-04/scene.npz`の動作記録をそのまま使い、
面ごとの材質を読み直して再描画しました。形状・法線・剛体の割り当て・動作記録は一致しています。
元の動作記録と旧描画は`bento-realistic-04/`に残しており、以前の整理記録はその中の`cleanup.json`です。

試作・失敗時の録画、旧版の動画、調査用の一時ファイル、インストールキャッシュは整理済みです。
連番画像はZIPに保管し、重複する展開済みフォルダを削除しました。
実行環境、最終版の実測姿勢・形状、動画、プレビュー、検証結果は残しています。

形状は[visuals.py](visuals.py)で手続き的に生成しています。RoboCasaの素材は取り込んでいません。
[RoboCasaの公式素材一覧](https://robocasa.ai/docs/build/html/assets/objects.html)には
ブロッコリーなどがありますが、炊いたご飯・卵焼き・唐揚げの一式は確認できませんでした。

## 起動

Isaac Sim / Isaac Lab / Arenaがインストール済みのコンテナ内で、リポジトリルートから実行します。
公式ロボット資産へのアクセスが必要です。CPU指定は物理計算用です。
GUI表示とRTX描画にはRTX対応GPUを使います。

この作業環境（`/work`）には、コンテナのローカル領域`/tmp/arena-runtime/venv`へ依存環境をコピーしました。
コピー時にNFS側の元の`.venv`と`data/arena-runtime`は削除・移動していません。
以下の起動コマンドはローカルコピーを使います。
次のランチャーは非root実行、EULA設定、専用領域に展開した共有ライブラリを設定します。

```bash
cd /work
/tmp/arena-runtime/run-python -m isaaclab_arena_examples.bento.simulation \
  --device cpu --headless --viz none \
  --output /tmp/arena-runtime/outputs/bento-tongs-result.json
```

環境の導入方式は既存[READMEのInstallation節](../../README.md#installation)を参照してください。
今回使用した配布パッケージ版のコマンドは、次のとおりです。

```bash
UV_PROJECT_ENVIRONMENT=/tmp/arena-runtime/venv /tmp/arena-runtime/bin/uv sync --frozen --no-default-groups --group isaaclab-from-wheel
```

これは再導入用のコマンドです。現在の環境では実行済みです。
Python 3.12も`/tmp/arena-runtime/python`へ導入しています。
再導入する場合は`UV_PYTHON_INSTALL_DIR=/tmp/arena-runtime/python`と
`UV_CACHE_DIR=/tmp/arena-runtime/uv-cache`を指定してください。

リポジトリ標準のArena開発コンテナでGUI表示する場合:

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh -m isaaclab_arena_examples.bento.simulation \
  --device cpu --viz kit --keep-open
```

起動後、1秒の静止を挟み、接近→下降→トングを閉じる→持ち上げ→移送→配置→開く→退避を実行します。
`--keep-open`を省略すると、判定後に終了します。

```bash
# 有限ステップの自動実行。成功は終了コード0、失敗は非0。
/isaac-sim/python.sh -m isaaclab_arena_examples.bento.simulation \
  --device cpu --headless --viz none \
  --output /tmp/yaskawa_bento_result.json

# シーンを動かさずに確認。--max-stepsは物理ステップ数。
/isaac-sim/python.sh -m isaaclab_arena_examples.bento.simulation \
  --device cpu --viz kit --inspect --max-steps 20000
```

GUIはウィンドウを閉じて終了できます。`--waypoint-timeout 20`で各移動の制限時間を変更できます。
既定の結果JSONは`/tmp/yaskawa_bento_result.json`です。保存するにはホストにマウントされた
パスを`--output`へ指定してください。JSONには成功/失敗理由、ステップ数、最終フェーズ、
食品の実測位置・XYZW姿勢・速度・解放状態・区画内判定を記録します。

## 画像収集と動画

ロゴ対応版の動画は`/tmp/arena-runtime/outputs/bento-logo/`に保存しています。
全体の動画と近接動画はそれぞれ960×540、30 fps、1,490フレーム、約49.67秒です。
物理時間138.38秒の動作を3倍速にし、最初に1.5秒、最後に2秒の静止画像を加えています。
米粒などの頂点色、ロボットの面ごとの材質色、実メッシュからの影を描画します。
今回のYASKAWAロゴは色を付けた形状としてUSDに含まれ、画像テクスチャは必要ありません。

A100で動くNVIDIA nvdiffrastで、実際のUSD形状と記録したPhysX姿勢から描画しています。
Isaac SimのRTXカメラ映像ではありません。

再作成する場合は、未使用の保存先を指定します。

```bash
cd /work
BENTO_CAPTURE_DIR=/tmp/arena-runtime/outputs/bento-tongs-new

# 1. 実シミュレーションの形状と姿勢を収集。
/tmp/arena-runtime/run-python -m isaaclab_arena_examples.bento.simulation \
  --device cpu --headless --viz none --capture-fps 10 \
  --capture-dir "$BENTO_CAPTURE_DIR" --output "$BENTO_CAPTURE_DIR/result.json"

# 2. 連番PNGを作成してMP4にまとめる。
/tmp/arena-runtime/run-python -m isaaclab_arena_examples.bento.render_video \
  "$BENTO_CAPTURE_DIR" --save-frames --width 960 --height 540 --fps 30

# 3. 食材を見やすい近接カメラ。別名の画像・動画へ保存。
/tmp/arena-runtime/run-python -m isaaclab_arena_examples.bento.render_video \
  "$BENTO_CAPTURE_DIR" --camera bento --width 960 --height 540 --fps 30

# 4. 元モデルのロゴが読める近接画像。
/tmp/arena-runtime/run-python -m isaaclab_arena_examples.bento.render_video \
  "$BENTO_CAPTURE_DIR" --camera robot --preview-only
```

`--capture-dir`は新規または空のフォルダである必要があります。
同じ物理記録から動画だけ作り直す場合は、2番目のコマンドを実行します。
`--fps 10`にすると動作部分を実時間の速度で再生します。
`--preview-only`で先頭・末尾のPNGだけを確認することもできます。

`scene.npz`にはメッシュと実測姿勢、`capture.json`には収集条件と物理シミュレーションの結果、
`video.json`には画角・再生速度・簡易照明などの動画条件を保存します。
この環境には追加依存のnvdiffrast 0.4.0を
[NVIDIA公式ソース](https://github.com/NVlabs/nvdiffrast/tree/253ac4fcea7de5f396371124af597e6cc957bfae)
から導入済みです。新しい実行環境でCUDA動画描画を使う場合も、この依存が必要です。

## ロボットとシーン

資産の相対パスは
`Robots/Yaskawa/Motoman Next/NHC12/NHC12_A00.usd`です。
Isaac Labの`ISAAC_NUCLEUS_DIR`を使い、既存ランタイムの資産バージョンを維持します。
手元の同機種USDは`--robot-usd /path/to/NHC12_A00.usd`で指定できます。
参照先のconfigurationやmesh等も解決できるよう、元のフォルダ構成を維持してください。

関節は`joint_1_s`、`joint_2_l`、`joint_3_u`、`joint_4_r`、`joint_5_b`、`joint_6_t`。
手先は`link_6_t`、トングの食品中心TCPはローカル`(0, 0, -0.20)` mです。
USDの`flange/tool0`はX軸180度回転しており、この符号に対応します。
作業台の上面と固定アームのベースは高さ0.75 mです。

食品と弁当箱は詳細な表示メッシュと単純な衝突形状を分けて生成します。
食品の寸法、色、質量、供給位置、配置先は[meal.py](meal.py)で変更できます。
改善版の食品の衝突形状は直方体です。各食品の表示メッシュはその寸法内に収めます。
台・箱の形状を変更するときは、`simulation.py`と食品の区画境界も合わせて変更してください。

## 物理と判定の範囲

- アームはCPU PhysXで動かし、減衰最小二乗IKで関節位置を制御します。
  TCPオフセットをJacobianに反映し、関節制限と1ステップの指令変化量を制限しています。
- 保持は理想的な固定関節による近似です。実測TCPが食品中心の目標から6 mm以内、
  姿勢誤差0.05 rad未満、手先リンクの並進速度0.035 m/s未満で、
  顎が設定幅まで閉じた状態を0.4秒維持して保持します。
  相対姿勢を保存して接続するため、食品を目標座標へ瞬間移動させる処理はありません。
  接続直前にも実測食品中心とTCPの距離を確認し、14 mmを超えた場合は失敗にします。
- 保持関節は食品をロボットのarticulationに組み込まず、解放後は重力と接触に従います。
  顎は表示形状であり、摩擦把持の成立・滑落・変形・破損・衛生条件はモデル化していません。
- 成功には全食品が解放済みで、回転を考慮した全寸法が割り当て区画内に収まり、
  底面が弁当床から8 mm以内にあることを要求します。
  並進速度0.02 m/s未満・角速度0.1 rad/s未満も全食品で0.5秒連続して確認します。
  移動タイムアウト、上限ステップ到達、解放後3秒以内に静止・配置できない場合は失敗です。
- これは固定の経由点を使うデモです。障害物回避の経路計画は実装していません。

## 検証状況

公式6.0/6.1のトップUSDを比較し、同一ファイルであることを確認しました。
以前のプリミティブ版ではUSDの関節フレームと制限から供給位置・配置先・上空への
到達可能性を小規模FK/IK計算で確認しました。この計算は衝突回避を評価していません。
改善版の食品寸法に更新した配置判定のpytestは9件すべて成功しました。

このDockerコンテナのA100-PCIE-40GB上で、描画なし・CPU PhysXによる盛り付けを実行しました。
改善版は下降の追従を修正した回で16,935ステップ、最終的なトング寸法と幅を含む回で
16,606ステップで成功しました。成功には全4品の解放、全寸法の区画内包含、床に近いこと、
0.5秒連続の静止が必要です。最終版の全食品は速度0で静止しました。
これは固定配置のデモの検証であり、接触による摩擦把持や異なる配置の成功を保証しません。

別担当が配置・開放中の10 Hz記録270フレームについて、両顎と箱の形状を確認しました。
外壁・仕切りへの貫通はなく、卵の配置中の1フレームで床と約0.012 mmの微小重なりがありました。
記録した時点の形状検査であり、フレーム間や接触力の検証ではありません。
配置判定のpytestは9件成功し、Black、isort、flake8、codespellの検証も通過しています。
最終版の2本のMP4は全フレーム再デコードで仕様を確認しました。
2つのZIPにはそれぞれ1,490枚のPNGがあり、全件のCRC検証が成功しています。

ロゴ対応の回帰検証では、頂点を共有する2つの四角形に白・青の材質を割り当て、
三角形への変換後も両色と形状・法線が保たれることを確認しました。
実際のUSDから元の青い材質を37,929頂点に取り込み、近接画像でYASKAWAの文字を確認しました。
描画データ更新前後の形状・法線の最大差は0で、全1,385時点の動作記録も一致しました。

- 結果: `/tmp/arena-runtime/outputs/bento-logo/result.json`
- 実行ログ: `/tmp/arena-runtime/logs/bento-realistic-04.log`

導入済み環境はPython 3.12.15、Isaac Sim 6.0.0.1、Isaac Lab 3.0.0b2、PyTorch 2.11.0+cu128です。
Arenaは`/work/isaaclab_arena`からimportされます。
仮想環境は約27 GiBです。再導入用uvキャッシュは整理済みで、再導入時は再取得します。
動画・画像・Python本体・仮想環境・キャッシュはコンテナ内ローカルの`/tmp/arena-runtime`に保存します。
NFS上の`data`には保存しません。
コンテナで不足していたGL/X11共有ライブラリは、`/tmp/arena-runtime/sysroot`へ展開して参照しています。
システムのPython環境やOSパッケージは変更していません。
未展開のIsaac Lab／Isaac-GR00T submoduleは変更せず、配布パッケージ版を導入しました。
A100はRT Coreを持たず、[Isaac Sim 6.0の公式対応GPU](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)
に含まれません。今回の描画なし物理実行は確認できましたが、GUI／RTX描画には対応GPUが必要です。

判定の回帰テストは、開発コンテナ内で次のように実行できます。

```bash
/tmp/arena-runtime/run-python -m pytest -q -p no:cacheprovider \
  isaaclab_arena_examples/tests/test_bento_placement.py
```

参照:
[NVIDIA Yaskawa assets](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_robots_manipulator.html#robots-manipulator-yaskawa)、
[Isaac Lab 3.0 beta2 IK tutorial](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/05_controllers/run_diff_ik.html)。
