# 移動ロボットによる図書館の返却・貸出デモ

4輪台車に載せたYaskawa NHC12アームで、次の順に本を扱います。

1. 返却ボックスからLIB-001を取り出し、本棚へ移動してA-01へ縦に収納する。
2. 返却ボックスへ戻り、LIB-002を取り出してA-02へ収納する。
3. 貸出予定のLIB-003をカタログで照合し、A-03から引き抜く。
4. 貸出ボックスまで運び、本を入れて解放する。

本棚は上段・下段と背景の本を持つ高さ2.1 mの書架です。対象本には色付きの背と
IDマーク、各棚位置には同色のマークを付けています。返却ボックスは茶色、貸出ボックスは青緑色です。
棚前で本を持ち上げて水平に抜き差しする経由点を使います。

本IDと棚位置は [scene.py](scene.py) の `CATALOG` で対応付けています。
探索は既知のカタログ情報による棚番の照合です。画像認識や未知の本の発見は行いません。
台車移動は指定経路に沿ってキネマティックな台車とアームのルート姿勢を更新する方式です。
車輪の駆動トルクや接地・スリップを評価する走行モデルではありません。
把持はトングの開閉表示と実測相対姿勢の固定拘束で近似し、本は剛体として扱います。

## 現在のコンテナで実行

新しいDockerコンテナは不要です。既存の非rootユーザーで実行してください。
この環境の `.venv/bin/python` はリンクが切れているため、残っているPython実体を直接使います。
依存環境の再インストールやリンク変更は行っていません。

```bash
cd /work
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export WARP_CACHE_PATH=/tmp/library-warp-cache
export XDG_CACHE_HOME=/tmp/library-cache
export LD_LIBRARY_PATH=/work/data/arena-runtime/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
LIBRARY_PYTHON=/work/data/arena-runtime/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
LIBRARY_CAPTURE=/tmp/library-mobile-new

"$LIBRARY_PYTHON" \
  -c 'import sys,runpy; sys.path.append("/work/.venv/lib/python3.12/site-packages"); runpy.run_module("isaaclab_arena_examples.library.run_library",run_name="__main__")' \
  --device cpu --headless --viz none --capture-fps 10 \
  --capture-dir "$LIBRARY_CAPTURE" --output "$LIBRARY_CAPTURE/result.json"

"$LIBRARY_PYTHON" \
  -c 'import sys,runpy; sys.path.append("/work/.venv/lib/python3.12/site-packages"); runpy.run_module("isaaclab_arena_examples.library.render_video",run_name="__main__")' \
  "$LIBRARY_CAPTURE"
```

`LIBRARY_CAPTURE` は新規か空の保存先を指定してください。共有キャッシュを使う場合は
実行ユーザーが書き込めることを確認してください。`--capture-dir` を省略すると物理実行のみです。
CPU指定は物理計算用です。GPUは既存コンテナから見える割り当てを使用します。
公式NHC12 USDへのアクセスが必要です。ローカル資産は `--robot-usd /path/to/NHC12_A00.usd`。
参照先のメッシュ等も元の構成で揃えてください。

標準Arena環境では `/isaac-sim/python.sh -m isaaclab_arena_examples.library.run_library` でも起動できます。
GUIはRTX対応GPUとディスプレイが必要です。`--headless --viz none` を `--viz kit --keep-open` へ
変更して表示します。`--inspect` は動作を開始せずシーンを確認します。
`--max-steps` の既定値は80000、`--waypoint-timeout` は30秒です。

## 成果物と判定

`result.json` は本ID・棚番、カタログ照合・把持・返却・貸出のイベント、
各本の目標位置と実測姿勢・速度・解放状態を保存します。
全3冊が解放済みで、目標から各軸8 mm未満、姿勢誤差0.1 rad未満、
並進速度0.02 m/s未満・角速度0.1 rad/s未満の状態を0.5秒維持し、
貸出対象のカタログ照合も完了した場合に成功します。成功は終了コード0です。

動画は実測PhysX姿勢とUSD形状を使ったCUDA描画です。RTXカメラ映像ではありません。
冒頭に全景、続いて台車を追う画角で返却・貸出を表示します。10 Hz記録を30 fpsで再生するため
動作部分は3倍速です。`library-mobile.mp4`、確認用PNG、描画条件JSONを生成します。
物理記録は `scene.npz`、記録条件と結果は `capture.json`。`--preview-only` でPNGだけを作れます。

以前の固定台版動画は `/work/outputs/library-video-20261019/` に残しています。

## 移動版の検証結果

2026-10-19、既存コンテナ内の非root実行・CPU PhysXで全工程が成功しました。
22,793ステップ（物理時間189.94秒）で完了し、3冊すべての解放・配置・静止を確認しました。
最終位置誤差は1.17〜1.87 mm、最終速度は全成分0です。
カタログ照合→把持→棚返却／貸出ボックス配置のイベントと、
台車の返却・棚・貸出停止位置を記録しています。
棚の上板下面は1.78 mとし、試験で見つかった手首と上板の干渉を解消しました。

足回りを含む1,901時点・全19工程を記録しています。未知の障害物回避、
画像認識、車輪の物理走行、摩擦把持、他の棚寸法や本配置は検証していません。
Docker新規作成、依存の再インストール、ホスト設定変更は行っていません。

作成済みの移動版動画は `/work/outputs/library-mobile-20261019/library-mobile.mp4` です。
960×540、30 fps、68.37秒（動作部分3倍速）です。同フォルダにプレビューPNG、
実測記録、結果JSON、描画条件とフレーム検証結果を保存しています。

## 密集した棚・傾いた本を扱う双腕試作

`crowded_shelf.py` は棚作業に絞った別の試作です。検証済みのNHC12を2本使い、
片方が返却本を保持し、もう片方の衝突付きの薄い本押さえで隣接本を押して隙間を広げます。
周囲の15冊も自由な剛体で、1冊は15度傾けて配置します。
隣接本を目標座標へ直接移す処理はありません。
返却本の把持は既存例と同様に理想的な固定拘束です。

上記の既存ランタイム・キャッシュ設定で実行します。

```bash
"$LIBRARY_PYTHON" \
  -c 'import sys,runpy; sys.path.append("/work/.venv/lib/python3.12/site-packages"); runpy.run_module("isaaclab_arena_examples.library.crowded_shelf",run_name="__main__")' \
  --device cpu --headless --viz none --output-dir /tmp/library-crowded-new

"$LIBRARY_PYTHON" \
  -c 'import sys,runpy; sys.path.append("/work/.venv/lib/python3.12/site-packages"); runpy.run_module("isaaclab_arena_examples.library.render_crowded",run_name="__main__")' \
  /tmp/library-crowded-new
```

`--no-capture --stop-after <工程名>` は録画なしで特定工程までを確認する診断用です。
診断完了は本の返却・貸出の成功を意味しません。失敗時は原因確認用の最終姿勢を1時点保存します。
接触操作は静摩擦0.12・動摩擦0.08の設定で試験します。
未知の摩擦、本の変形、任意の倒れ方や完全に崩れた山を扱えることは保証していません。

床のちらつき・動く帯の修正動画は
`/work/outputs/library-shadow-fixed-20261019/library-mobile.mp4` です。
重複する床面による深度競合を除き、影の投影範囲をワールド座標に固定しました。

密集棚の途中検証では、接触で隙間を約25 mmから99 mmへ広げ、
35 mm厚の返却本を挿入・解放し、両腕が棚から退避するところまで確認しました。
双腕の通し記録は `/work/outputs/library-crowded-dual-20261019/` に保存しました。
`crowded-shelf.mp4` は返却ボックスからの取り出し、隙間拡大、挿入、隣接本の支持、
貸出本の取り出しとボックスへの配置を含みます。固定台に載せた2本のアームの試作で、
移動台車との統合は含みません。

22,794ステップ（189.95秒）の実測記録では、隙間は25.0 mmから99.0 mmへ広がりました。
本の実測姿勢から上端と支持高さを計算し、上から押さえを入れます。
支持しても隣接本が横倒しになる場合があり、棚全体の倒れ防止は未実装です。
補助アームの台は1.10 m、天板下面は2.15 mで、前腕の接触を避けています。

返却本は周囲の接触で横に17.53 mm、奥行き2.44 mm移動して静止しました。
当初の各軸15 mm判定では不合格でした。横方向20 mm・奥行きと高さ15 mmを
指定スロットの許容とし、変更していない実測記録で再判定しています。
元の結果は `result-original-15mm.json`、再判定の条件・各判定結果は `result.json` です。
貸出本のボックス内配置、両対象本の解放・静止、隙間60 mm超、支持工程の実行も確認しました。
新しい許容値での物理実行は繰り返していません。把持は理想拘束で、任意の崩れ方への対応は未検証です。
