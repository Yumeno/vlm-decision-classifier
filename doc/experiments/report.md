# 実験レポート — ローカルVLM選択式画像分類デモ

## 1. 概要

本デモは、Jev的な選択式判定(回答ラベルの `logprobs` から候補間の相対スコアを読む方式)を、マルチモーダル入力・任意のローカルVLMに適用したときの挙動を確かめる技術検証である。分類の精度そのものを追い込むことや、メタデータ補助による精度向上は目的にしていない。

専用に新規生成した画像データセット(元画像31枚、メタデータ除去コピー4件の計35ケース)を使い、通常版Qwen3.5 9B GGUFとGemma 4 12B GGUFの2モデルで、選択式方式(`choice`)と通常のJSON生成方式(`json`)を同一条件で対応比較した。Qwenについては、`taxonomy` の `none` 選択肢の文言修正前後(0.4.0→0.4.1)の再評価と、llama.cppの画像キャッシュ改造の有無(未改造/改造)による速度比較も行った。

以下の数値は各実験フォルダの `summary.md`・`cases.csv`・`run.json`(E5は `vanilla.json`/`patched.json`)を一次資料として転記したものである。元画像N=31件と小標本のため、1〜2件の差は誤差の範囲として扱い、強い結論は書かない。精度の数値は「選択式という方式が実際にどう振る舞うか」を見るための材料であり、改善余地の列挙が目的ではない。

## 2. 条件

| 項目 | 値 |
|---|---|
| データセット版 | v1.0.0(元画像31枚 + メタデータ除去コピー4件 = 35ケース。`dataset/manifest.jsonl` SHA256 `941faf84f0c1e3f027748766b6ce2ec4181cc8749b0c1ad70ec8113b528baab0`) |
| taxonomy | 0.4.0(E1/E1-J/E2/E2-J、SHA256 `ec7b2a2f750c7848c56033c9b4a5aee799df1aa7548c126528e6cfc1db4b26aa`)→ 0.4.1(E1b以降、SHA256 `6f7038cbd525af1552af1c95158090ebc38a1cd1303191af8cff8c7fdfb7564f`)。0.4.1は複数選択軸(`outfit`/`character`)の `none` 選択肢に説明文(`none_criteria`)を足し、`other_original` の説明を明確化しただけで、**正解ラベルは変えていない** |
| Qwenモデル | `Qwen/Qwen3.5-9B` 公式対応。GGUF: `lmstudio-community/Qwen3.5-9B-GGUF`(revision `1379f25c6b505a3fc737bd7818cb09389cf807c1`)`Qwen3.5-9B-Q4_K_M.gguf` SHA256 `cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13`。mmproj `mmproj-Qwen3.5-9B-BF16.gguf` SHA256 `330d17547bfdbcd0e7a0cb3f4b06b4ceeac0aaa1e449122d9d66ef957aeb74b3` |
| Gemmaモデル | `google/gemma-4-12B-it` 公式対応。GGUF: `lmstudio-community/gemma-4-12B-it-GGUF`(revision `40b870babe39398ce917bbf4b4ff9b5a1f12710e`)`gemma-4-12B-it-Q4_K_M.gguf` SHA256 `95d83ba36642b1f385fb906b5962a71763361be3bac930a709945f72d97473f8`。mmproj `mmproj-gemma-4-12B-it-BF16.gguf` SHA256 `7fd884a5f7d9ee60f6b88e5b51287151e547454b2b60a4a6270bb5434310ad0e` |
| サーバー(LM Studio) | E1/E1-J/E2/E2-J/E1b/E2b。エンジン `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.41.0`、LM Studio CLI commit `69d945a` |
| サーバー(llama-server) | E3/E4/E5。上流 `https://github.com/ggml-org/llama.cpp` commit `f95b0d95394d5e311ba8228689972843178c5e28`(`llama-server --version` は `0.4.1-dev (build 1, commit f95b0d9)`)。未改造版(E3)と改造版(E4)は同一コミット・同一ビルド手順から作り、差はパッチ1行のみ |
| パッチ | `doc/patches/llamacpp-mtmd-checkpoint.patch`(`tools/server/server-context.cpp` の `do_checkpoint = do_checkpoint && !has_mtmd;` を無効化) |
| GPU | RTX 3090 24GBのみで速度計測。E1/E1-J/E2/E2-Jは別プロジェクトのパッチ適用版llama-serverが常駐した状態で計測(`run.json` に記録)。E1b/E2b以降はRTX 3090に他プロセスなしを確認して計測 |
| クライアント設定 | `max_edge=1024`、`temperature=0`、`top_logprobs=20`、`reasoning_effort="none"`(未指定だとthinkingが先に出て失敗する) |
| ツールのコミット(`tool_commit`) | E1/E2: `5f205960d1ed9d8c49dd2b528d0382f425809b6d`(dirty=true、評価直前に実行環境JSONへ`gpu_used`等を書き足したため。コード自体は変更なし) / E1b/E2b: `39285e41bdb4aeae9f2e6525c5c9933f60da60ed` / E3: `c40f0570982fa6a0d73273aadc73964c1b4bc31a` / E4: `847070af31f8302585454fbe35218b06f3ed8dc8` / E5未改造: `e9058aaac65b2497d6551c1baed29264f6a148f7` / E5改造: `c40f0570982fa6a0d73273aadc73964c1b4bc31a`(いずれも dirty=false) |

## 3. 実験一覧

| 実験ID | モデル/サーバー | 質問方式 | taxonomy | フォルダ |
|---|---|---|---|---|
| E1 / E1-J | Qwen3.5 9B / LM Studio | choice / json | 0.4.0 | [`E1_qwen-lmstudio/`](E1_qwen-lmstudio/summary.md) |
| E2 / E2-J | Gemma 4 12B / LM Studio | choice / json | 0.4.0 | [`E2_gemma-lmstudio/`](E2_gemma-lmstudio/summary.md) |
| E1b | Qwen3.5 9B / LM Studio | choice / json | 0.4.1 | [`E1b_qwen-lmstudio/`](E1b_qwen-lmstudio/summary.md) |
| E2b | Gemma 4 12B / LM Studio | choice / json | 0.4.1 | [`E2b_gemma-lmstudio/`](E2b_gemma-lmstudio/summary.md) |
| E3 | Qwen3.5 9B / llama-server 未改造 | choice | 0.4.1 | [`E3_qwen-llamaserver-vanilla/`](E3_qwen-llamaserver-vanilla/summary.md) |
| E4 | Qwen3.5 9B / llama-server 改造 | choice | 0.4.1 | [`E4_qwen-llamaserver-patched/`](E4_qwen-llamaserver-patched/summary.md) |
| E5 | Qwen3.5 9B / llama-server 未改造・改造 | 同一画像・質問替え、A→B→A | 0.4.1 | [`E5_qwen-llamaserver/vanilla.json`](E5_qwen-llamaserver/vanilla.json) / [`patched.json`](E5_qwen-llamaserver/patched.json) |

## 4. 結果

### 4.1 精度(7軸、元画像N=31件)

主な結果は taxonomy 0.4.1 のE1b/E2b。E1/E2(0.4.0、`none` の文言修正前)は参考として併記する。数値は各 `summary.md` からの転記(百分率、小数第1位)。

| 軸 | Qwen E1(choice/json) | Qwen E1b(choice/json) | Gemma E2(choice/json) | Gemma E2b(choice/json) |
|---|---|---|---|---|
| image_type | 96.8 / 93.5 | 96.8 / 96.8 | 100.0 / 90.3 | 100.0 / 90.3 |
| art_style | 77.4 / 74.2 | 77.4 / 77.4 | 83.9 / 80.6 | 83.9 / 80.6 |
| color | 100.0 / 90.3 | 100.0 / 93.5 | 100.0 / 93.5 | 100.0 / 93.5 |
| subject | 96.8 / 90.3 | 96.8 / 93.5 | 96.8 / 90.3 | 96.8 / 90.3 |
| situation | 80.6 / 80.6 | 80.6 / 83.9 | 80.6 / 67.7 | 80.6 / 67.7 |
| outfit(完全一致) | 77.4 / 87.1 | 74.2 / 90.3 | 71.0 / 80.6 | 71.0 / 83.9 |
| character(完全一致) | 64.5 / 87.1 | **93.5** / 87.1 | 83.9 / 83.9 | **93.5** / 93.5 |

taxonomy 0.4.1(E1b/E2b)で選択式のcharacterが両モデルとも93.5%に上がった。内訳は `other_original`(Alisa・second_originalでない人物)のrecallで、Qwen 0/11→9/11、Gemma 8/11→9/11(候補別TP/FP/FNは各 `summary.md` の character セクション)。E1では選択肢の文言が「none of the above」で、設計上「人物なし」に使うつもりの`none`を「上のどれでもない」と読めてしまい、`other_original`が一度も選ばれなかった(Qwen)。0.4.1は`none_criteria`の説明文を足しただけで正解ラベルは変えていないため、この変化は選択肢の文言設計の影響として読む(方式そのものの限界ではない)。

outfit・situationなど他の軸は0.4.0→0.4.1でおおむね同水準か1〜2件の変動で、taxonomy変更よりも実行時の揺れ(平均リクエスト数がQwen 11.19→11.35、Gemma 10.19→9.97に変化)の範囲として扱う(要因は切り分けていない)。

### 4.2 outfitの既知の弱点(選択式の school_uniform 誤検出)

選択式の複数選択軸`outfit`では、Alisaの制服(ベスト+ペンシルスカート、正解`office_wear`)に`school_uniform`が追加で付く誤りが残った。E1b/E2b(choice)の元画像ケースで確認したケースIDは以下の通り(`cases.csv` の `pred_outfit` 列より):

- Qwen(E1b): A02、A03、A04、N01、N02、N03(6件)
- Gemma(E2b): A02、A03、A04、M02、N01、N02、N03(7件、うちA02は`sailor_uniform`も付く)

候補別の精度では`school_uniform`のPrecisionが低い(E1b: 0.54、E2b: 0.46。Recallはどちらも1.00で見逃しはない)。これは複数選択の確認(yes/no)段階で近い候補も「はい」と判定されやすいことを示す。ほかにoutfitが空(none)になる件が各モデル1件あった(Gemma M01、Qwen O06。`cases.csv`で確認)。

`school_uniform`と`office_wear`を相互排他と明記する説明文の案(0.4.2)も検討したが、作者の判断で**変えていない**(2026-09-27)。理由は実験条件を変えないため、また評価セットを見てからの事後調整になるため。この誤りは選択式の弱点としてそのまま報告する。

### 4.3 メタデータ証拠とメタデータ除去コピー

メタデータ(LoRA指定・トリガーワード・形式)の集計は全35ケースを分母とし、E1/E1b/E2/E2bのいずれも format・characters・loras・trigger_words の完全一致率は100%(35/35、`summary.md` の「メタデータ」節)。

元画像(A01〜A04)とメタデータ除去コピー(A01-strip〜A04-strip)の character 判定を比較すると、全実験・全モードで予測ラベルが一致した(`summary.md` の「派生ケース」表)。誤りが出たケース(E1b の json、A04: `other_original`)も元・派生で同じ誤りを再現しており、メタデータの有無が画素からの character 判定を変えていないことを示す。

### 4.4 処理時間

1画像あたりの `classification_wall_ms`(元画像N=31件、**失敗ケースも含む全件**の値。単位ms)。成功ケースのみの分位値は各 `summary.md` に併記している。

| 実験 | mean | p50 | p90 | 平均リクエスト数 | 失敗 |
|---|---:|---:|---:|---:|---|
| E1 choice(Qwen/LM Studio) | 10264.5 | 10165.8 | 13014.7 | 11.19 | なし |
| E1b choice(Qwen/LM Studio) | 9280.3 | 9257.8 | 11530.3 | 11.35 | なし |
| E3 choice(Qwen/llama-server未改造) | 8706.5 | 8735.8 | 10781.1 | 11.35 | なし |
| E4 choice(Qwen/llama-server改造) | 3127.5 | 2986.0 | 3756.4 | 11.35 | なし |
| E2 choice(Gemma/LM Studio) | 6396.7 | 6380.3 | 7729.9 | 10.19 | candidate_confirmation_error 2件 |
| E2b choice(Gemma/LM Studio) | 5449.8 | 5457.5 | 5914.7 | 9.97 | なし |
| E1 json(Qwen/LM Studio) | 2347.9 | 2224.3 | 2464.4 | 1.13 | json_format_error 2件(形式不正 6.5%) |
| E1b json(Qwen/LM Studio) | 1964.3 | 1925.3 | 2090.6 | 1.06 | json_format_error 1件(形式不正 3.2%) |
| E2 json(Gemma/LM Studio) | 2472.4 | 2283.5 | 2504.0 | 1.13 | json_format_error 2件(形式不正 6.5%) |
| E2b json(Gemma/LM Studio) | 2255.3 | 2069.3 | 2278.1 | 1.13 | json_format_error 2件(形式不正 6.5%) |

E1/E2はサーバー(LM Studio)にほかプロジェクトのllama-serverが常駐していたのに対し、E1b以降はRTX 3090に他プロセスがない状態で測っている。処理時間の短縮(Qwen choice 10.3→9.3秒、Gemma choice 6.4→5.4秒)は、この条件差とtaxonomy変更による確認回数の変化の両方が影響しうるため、要因は切り分けていない。

### 4.5 キャッシュ改造(E3 vs E4、E5)

同一llama.cppコミット・同一ビルド手順・起動引数で、パッチ1行の有無だけを変えたE3(未改造)とE4(改造)を比較した。

- 1件あたりの処理時間: 8.7秒(未改造)→ 3.1秒(改造)、約64%短縮。
- 予測結果: E3とE4は`cases.csv`の`pred_*`列で全35ケース・全軸を突き合わせ、完全に一致することを確認した(判定は変わらない)。E1bとE3も同様に全35ケース・全軸一致。

E5(同一画像で質問だけ変え、各繰り返しの頭で別画像Bを1回送る小ベンチマーク。`vanilla.json`/`patched.json`):

| | 未改造 | 改造 |
|---|---:|---:|
| 1軸目の中央値(n=3) | 749.5 ms | 327.8 ms |
| 2軸目以降の中央値(n=18) | 653.7 ms | 196.0 ms |
| A→B→Aの最上位ラベル不一致 | 0/21組 | 0/21組 |
| A1とA2の相対スコアの最大絶対差(全体) | 0.0170 | 0.0144 |

未改造版は同じ画像に質問だけ変えても毎回ほぼ同じ時間がかかる(上流issue [#26994](https://github.com/ggml-org/llama.cpp/issues/26994) の報告どおり、画像の再エンコードが起きている)。改造版は1軸目(画像切替直後)も速い(328ms)。これは起動引数`-np 4`でスロットが4つあり、画像Bを送ってもAの状態が別スロットに残っていて、サーバーが「内容が最も近いスロット」を選んで再利用するため(サーバーログの`selected slot by LCP similarity`)。E3/E4の全件評価も同じ`-np 4`構成なので、この挙動込みの実測値として扱う。A1/A2間のスコア差(最大0.017)は未改造版にもあり、パッチ由来ではない。

E1bとE3の差(9.3秒→8.7秒)はLM Studioとllama-serverというサーバー実装の違いであり、パッチの効果には含めない。

## 5. 失敗と試行錯誤

- **`none`の選択肢の文言**: E1/E2では「none of the above」という文言が「人物なし」ではなく「上のどれでもない」と読める設計になっており、選択式のcharacterで`other_original`が一度も付かなかった(Qwen)。`none_criteria`の説明文を足す修正(taxonomy 0.4.1)で改善したが、正解ラベルは変えていない。
- **Gemmaの Channel Error(E2のみ)**: yes/no確認2件(G03・G05)がLM Studioサーバー側の`Engine protocol predict request failed: fetch failed`で失敗した。評価器は失敗として分母に残し、E2bでは再発しなかった。
- **JSON形式不正の中身**: 単一選択の軸をリストで返す誤りがあった(E1 の Qwen: O01 の situation、O06 の subject)。
- **`--kv-unified`**: 最初`--kv-unified`なしで起動したところ`n_ctx_slot = 2048`(スロットごとに分割)になり、LM Studio(コンテキスト8192の共有)と条件が変わってしまうため、測定前に停止して付け直した。
- **`-mg`の番号**: `-mg`はCUDAの並びで指定するが、`nvidia-smi`の番号とは逆順だった(この環境ではCUDA0 = RTX 3090)。`llama-server --list-devices`で確認して対応した。
- **スロット再利用**: 改造版は`-np 4`のスロットが複数あるため、画像を切り替えても別スロットに残った古い画像の状態を再利用できてしまう。E5テスト1の「1軸目=再エンコードあり」という前提は、この構成の改造版には当てはまらないと分かった。E3/E4と同じ条件での実際の挙動として、そのままE5の結果に記録している。

## 6. 限界

- 元画像31枚の小標本。単一軸の候補には1〜3件しかないものがある(`comic`・`ui`・`pixel_art`・`grayscale`・`line_art`等)。候補別の成績は個別例として読み、強い結論にしない。
- 自作データセット(Anima base v1.0 + 自作LoRA)であり、他のモデル・画風への一般化は確認していない。
- taxonomyは評価セットの生成・正解付与後に0.4.0→0.4.1へ調整している(`none`選択肢の説明文追加)。正解ラベルは変えていないが、評価結果を見てからの文言調整である点は限界として明記する。
- 処理時間は別プロセスの影響を受けうる。E1/E1-J/E2/E2-Jは、別プロジェクトのパッチ適用版llama-serverがGPUに常駐した状態で測定しており(`run.json`の`concurrent_processes`に記録)、E1b以降の測定(RTX 3090に他プロセスなし)と単純比較できない。

## 7. 追加課題(未実施)

初版MVPの完了条件には含めない。実施する場合は優先度を見て別実験として着手する。

- **E6**(説明文付きJSON生成との比較): 元プロジェクトでは実施済みだが、優先度は低く初版の完了条件に含めない。
- **E7**(束ね質問): 単独質問を初版の基準とし、束ね質問は比較実験として後回しにする。
- **E8**(`vision_only` 対 `metadata_assisted`、メタデータ補助の寄与を見る対照実験): 2026-09-28、作者の判断で追加課題とした。本デモの目的は選択式判定の技術検証であり、精度を上げるための補助情報の導入(メタデータ補助)は目的にしていないため。
