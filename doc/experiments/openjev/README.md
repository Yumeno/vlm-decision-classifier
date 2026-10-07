# OpenJev(llama-server の `/v1/systemone`)の比較と、llama.cpp b11447 への更新確認

2026-10-07 の記録。**各条件は1回の測定、元画像31枚の小標本**なので、少数の差から強い結論は出さない。スコアは候補内で割り直した**相対スコア**(正答確率ではない)。数値の出典は、このフォルダの `default/`・`criteria/` の `run.json` と `summary.md`(個人パス除去済み)、および DiffusionGemma 側の [`../dgemma/followup-12.md`](../dgemma/followup-12.md) §6。

## 1. 目的

上流 llama.cpp の PR ggml-org/llama.cpp#29818(2026-10-02 merge)で、llama-server に判定モデル向けの `/v1/systemone` が入った(対象: laya、julia-1、lev、openjev、kev)。画像つきで使えるのは OpenJev。本リポジトリの `dgemma_choice` モードのクライアントがそのまま使えるかを確かめ、同じ31枚・同じ taxonomy で DiffusionGemma と比べる。あわせて、llama.cpp を新しい版に上げても既存の結果が変わらないかを確認する(§7)。

## 2. 条件

- 重み: `ggml-org/OpenJev-GGUF` rev `10840f375658dea7afc5ff4711127bca8218b560`
  - `OpenJev-Q4_K_M.gguf`(18,973,872,288 B、sha256 `38b512277edaeec6cd251d146bff6d97bc328fa9db19755401d6e9ab1ccbcba5`)
  - `mmproj-OpenJev-Q8_0.gguf`(629,247,232 B、sha256 `e372cdbf59fdd6bd2504cb64c988b31c7a42ac406a8f711df4b7a7acd9216f1e`)
- **ライセンス**: 重みは CC BY-NC 4.0(ベースは `openjev/openjev`)。非商用。**重みは本リポジトリに含めない**。商用で使う場合は各自ライセンスを確認する。
- ランタイム: 未改造(vanilla)の llama-server b11447(commit `da263e7`)。ビルドは [`../phase4-runbook.md`](../phase4-runbook.md) と同じ設定(CUDA 12.8、`CMAKE_CUDA_ARCHITECTURES` 86;89、MSVC 19.44.35228.0)。
- 起動: `-ngl 99 -c 16384 -np 16 --kv-unified -sm none -mg 0`、RTX 3090 のみ(KV を含めて VRAM 約22.2GB)。1枚の probe では `-np 8` で約4.5秒、`-np 16` で約3.4秒だった(繰り返しても同じ値)。
- 評価: 既存の `dgemma_choice` モードを、`--dgemma-url` に llama-server を指して使った(クライアントは無変更。llama-server は余分なフィールド(samples・seed・template など)を無視する)。`--model openjev`、元画像31枚+派生4件、送信画像は JPEG(quality 90)・長辺1024、`--warmup 0`、`--prime` なし。質問文は `default` と `--dgemma-catchall-style criteria` の2通り。
- 比較対象: DiffusionGemma(vLLM、`GPU_MEM_UTIL=0.88`、画像トークン 140、seed 0)の同名条件([`../dgemma/followup-12.md`](../dgemma/followup-12.md) §6)。

## 3. 結果(元画像31枚、各1回)

| 条件 | 単一5軸(/155) | キャラ完全一致 | 服装完全一致 | キャラ TP/FP/FN | 服装 TP/FP/FN | 失敗 | 時間 mean / p50 ms |
|---|---|---|---|---|---|---|---|
| openjev default | 144 | 20/31 | 17/31 | 17/1/10 | 18/13/8 | 0 | 3346 / 3433 |
| openjev criteria | 144 | 23/31 | 21/31 | 20/1/7 | 23/9/3 | 0 | 3454 / 3550 |
| dgemma criteria s0 | 143 | 29/31 | 22/31 | 26/1/0 | 25/9/1 | 1 | 453 / 422 |
| dgemma default s0 | 143 | 27/31 | 21/31 | 24/1/3 | 25/13/0 | 1 | 452 / 421 |

OpenJev の単一選択の軸別(default・criteria とも同じ): image_type 31/31、art_style 26/31、color 31/31、subject 29/31、situation 27/31。

OpenJev の外れ(元画像):

- キャラの FN はすべて「他のオリジナル」(`other_original`)。default は G05、G07、M02、M03、O02〜O07、criteria は G07、M02、M03、O02、O03、O05、O07。
- 服装の外れ: default は G05・G07・M03・O06・O07 が other、M01 が school_uniform、N01・N02 が office_wear。criteria は M01 が school_uniform、N01・N02 が office_wear。

## 4. 分かったこと(小標本の範囲で)

- 単一選択5軸の合計は、OpenJev が 144/155 で、この比較では最も高かった(DiffusionGemma は 143)。差は1で、揺れの範囲。
- キャラの外れは「他のオリジナル」の取りこぼしが中心(FN 10 → 7)。FP は両条件とも1件。DiffusionGemma(キャラ 27〜29/31)には届いていない。
- 質問文を `criteria` にすると、OpenJev でもキャラ(20 → 23/31)・服装(17 → 21/31)が改善した。DiffusionGemma で見えた傾向と同じ向き(各1回なので、効果の大きさは言えない)。
- 時間は OpenJev が約3.3〜3.5秒、DiffusionGemma が約0.45秒で、約7.5倍遅い。画像トークンを 280 や 140 に揃えても約2.6秒(約5.8倍)で、精度もほぼ変わらない(§5)。調査メモによると OpenJev は質問ごとに1回ずつ forward pass を行う実装(今回は16問)。
- 失敗0件。同じ入力を繰り返したときの値も同じだった(1枚の probe)。
- OpenJev のモデルカードによると、主にスクリーンショットで訓練された判定モデルで、アニメイラストは主な訓練分布の外にある。キャラや服装の外れをこの点に結びつけるのは推測で、検証していない。

## 5. 画像トークン数(`--image-max-tokens`)

上の比較で画像トークンの設定が揃っていなかったので、llama-server の `--image-max-tokens` を変えて `criteria` の質問文で測り直した(b11447 vanilla、`-np 16 -c 16384 --kv-unified`、RTX 3090、各1回、`criteria_img280/`・`criteria_img140/` に記録)。

| 画像トークンの上限 | 1問あたりのプロンプト(llama-server ログの n_tokens) | 単一5軸(/155) | キャラ完全一致 | 服装完全一致 | キャラ TP/FP/FN | 服装 TP/FP/FN | 時間 mean / p50 ms |
|---|---|---|---|---|---|---|---|
| 既定(llama.cpp の Qwen-VL 既定、約1,000) | 約1,130(1,130〜1,139) | 144 | 23/31 | 21/31 | 20/1/7 | 23/9/3 | 3454 / 3550 |
| 280 | 362 | 142 | 22/31 | 21/31 | 19/1/8 | 23/9/3 | 2606 / 2594 |
| 140 | 227 | 143 | 22/31 | 22/31 | 19/1/8 | 24/9/2 | 2610 / 2597 |

- 画像 A01(16問)の `usage.input_tokens` は、既定 18,105、280 で 5,817、140 で 3,657。
- llama-server は「Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks」という警告をログに出す(上限を下げた条件)。
- 精度はほぼ変わらない(単一5軸 142〜144、キャラ 22〜23/31)。時間は既定より約25%短いが、280 と 140 で同じ。入力トークンが約1/3になっても変わらないので、1問ごとの forward pass(27B の密なモデル)が時間の大半を占めると推測している(未検証)。
- 画像トークンを揃えても、時間は DiffusionGemma の 453 ms の約5.8倍、キャラは 22 対 29/31。

## 6. 限界

- 各条件1回、31枚。数件の差は揺れの範囲かもしれない。
- §3 の表は画像トークンの設定が違う(DiffusionGemma は max_soft_tokens 140、OpenJev は llama.cpp の既定)。揃えた条件は §5 の「画像トークン数」で測った(各1回)。
- 画像トークンを揃えた測定(§5)も各1回。上限を下げると Qwen-VL の推奨(1,024 以上)を下回る。
- 時間は実装(llama-server `-np 16` と vLLM)が違い、画像キャッシュの状態や並列設定も同一ではない。
- 質問ごとの forward pass は調査メモの記述で、実装までは精査していない。

## 7. llama.cpp b11447 への更新確認

2026-10-07 に実施。

- 上流タグ b11447 = commit `da263e7275dfbaeefcd61504eaa4fd5247540e11`(現行の `f95b0d9` から347コミット後)。
- vanilla とパッチ版を、[`../phase4-runbook.md`](../phase4-runbook.md) と同じ設定でビルドした。パッチ [`../../patches/llamacpp-mtmd-checkpoint.patch`](../../patches/llamacpp-mtmd-checkpoint.patch) はそのまま適用でき(hunk は +270 行ずれ)、`llama-server --version` は `0.6.0-dev (build 1, commit da263e7)`。
- パッチが無効にする行(`do_checkpoint = do_checkpoint && !has_mtmd;`)は上流に残っており、issue ggml-org/llama.cpp#26994 も open のまま。**パッチは引き続き必要**。
- 退行確認: Qwen3.5 9B Q4_K_M、パッチ版の旧(f95b0d9)と新(b11447)を、5枚(A01、C01、G05、M02、O03)× モード choice・bundled × 各2回(2回目は画像キャッシュ済み)、JPEG 長辺1024、RTX 3090 で比較した。
  - 回答ラベルは 20/20 で一致。相対スコアの差の最大は 0(同一)。
  - 平均時間(旧 → 新):

| 条件 | 旧 f95b0d9 (ms) | 新 b11447 (ms) |
|---|---|---|
| choice 1回目 | 1969 | 1947 |
| choice 2回目 | 1261 | 1232 |
| bundled 1回目 | 714 | 704 |
| bundled 2回目 | 468 | 450 |

- 5枚×2回の小さな確認。これを受けて、新規セットアップの推奨版を b11447 に切り替えた([`../phase4-runbook.md`](../phase4-runbook.md)、[`../../reproduce.md`](../../reproduce.md))。過去の結果はすべて f95b0d9 で測ったもので、書き換えていない。
