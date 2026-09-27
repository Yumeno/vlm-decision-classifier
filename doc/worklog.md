# 作業記録

新しい順。compact 後の文脈復元用。詳細は各 PR と `doc/` を参照。

## ★ 現在地と引き継ぎ(2026-09-27 更新。compact 後はまずここを読む)

**状態**: E1b/E2b の実行前で待機中(作者が VRAM を別用途で使用中)。作者の合図があれば、手順書 `doc/experiments/phase3-runbook.md` どおりに実行する。

**リポジトリ**
- main には PR #1〜#6 が merge 済み(分類コア・評価器・データセット v1.0.0 まで)。
- 作業ブランチ `phase3/runbook`(push 済み、PR 未作成)には、手順書、実行環境 JSON、E1/E2 の結果(`doc/experiments/E1_qwen-lmstudio`・`E2_gemma-lmstudio`)、taxonomy 0.4.1(none_criteria)が入っている。**E1b/E2b の結果を足してから PR #7 にする**(並行 PR は積まない)。
- メインの作業フォルダ(`<repo>`)は `phase3/runbook` をチェックアウト済みで、`.venv`(py3.12)がある。`dataset/staging/`(管理外)には生成画像・検収用ページの素材・正解シートの読み出しが残っている。
- 使い終わった worktree が3つ残っている(`.claude/worktrees/agent-*`。phase2a・phase2b・generation-scripts)。どれも merge 済みなので削除してよい。

**E1b/E2b の実行コマンド**(GPU を使う前に作者の確認を取る。`lms ps` で他のモデルを確認し、アンロードは作者の了承後)
1. `lms load qwen3.5-9b --identifier qwen3.5-9b -y` → `nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv` で GPU を確認し、runtime JSON の記録を更新する(前回は RTX 3090 = bus 0D)。
2. `.venv/Scripts/python.exe -m classifier_demo probe --model qwen3.5-9b` → M01 で試打(`classify ... --mode choice/json`。`results/trial` を先に作る)。
3. `.venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --model qwen3.5-9b --modes choice,json --warmup 1 --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/qwen-lmstudio.json --runtime-label qwen-lmstudio-tax041 --note "E1b: taxonomy 0.4.1 (none_criteria)" --output-dir results/E1b_qwen-lmstudio`
4. `lms unload qwen3.5-9b` → Gemma で同じ手順(`gemma-4-12b-it`、`results/E2b_gemma-lmstudio`、runtime は gemma-lmstudio.json)→ アンロード。
5. `run.json`・`cases.csv`・`summary.md` を `doc/experiments/E1b_…`・`E2b_…` にコピーし(個人パスがないか確認)、E1/E2 との比較(特に character の other_original の recall、outfit の school_uniform の誤検出)を worklog に書いて、PR #7 を出す。

**これまでの主な結果(E1/E2、元画像31件)**: 単一選択の軸は、選択式が JSON 方式と同等以上だった。選択式の弱点は character の other_original の取りこぼし(Qwen 0/11)で、原因は「none of the above」の文言の解釈。0.4.1 でこれを直した。処理時間は、選択式が Qwen 10.3秒 / Gemma 6.4秒、JSON が約2.3〜2.5秒。

**残タスク(初版 MVP)**
- E1b/E2b → PR #7。
- Phase 4: llama.cpp の画像キャッシュ改造の比較(E3/E4/E5)。改造版は `~/Desktop/llamacpp-build/llama.cpp`(上流 `f95b0d9` + 1行パッチ、別プロジェクトが使用中)にある。未改造版は、同じコミットを別の場所にクローンしてビルドする。GPU とビルド作業の前に作者の確認を取る。
- Phase 5: レポート(`doc/experiments/report.md`)、README の実行手順、note 記事の下書き、公開前チェック(ライセンスは作者が決める)。
- 小さな修正: `classify --output` が出力先のフォルダを作らない。
- 追加課題(初版の完了条件外): E6(説明文付き JSON)、E7(束ね質問)。

**運用ルール(CLAUDE.md / メモリにもある)**: Sonnet が実装し、Codex(gpt-6-luna)がレビュー(5ラウンドで収束しなければ作者を呼ぶ)。VRAM を使う前に作者を呼ぶ。区切りごとに worklog に書く。コンテキストが 75〜85% になったら待機する。

**artifact**(非公開): 狙い一覧 (非公開の作業用ページ) 、正解付与シート (非公開の作業用ページ) (db の `labels` コレクション。最終版は `dataset/labels/labels_final.json` に固定済み)。

## 2026-09-25 — Phase 1 完了: データセット v1.0.0 の固定

**やったこと**
- 作者の検収で6枠を除外し、作者の指示で2回目を外部ツールで作った(O06・O07: 編集でグレースケール化・線画化、O04・O02: 背景と衣装の編集、G02: 新規生成、G07: テキストネーム→画像の2段階)。O07 は Codex の安全判定で出力されず、作者の指示で Antigravity に依頼した(2b)。
- 正解付与シート(artifact、db 機能で保存。狙いを初期値として表示)を作り、作者が31枠を確定した。5枠が規則とぶつかっていた(人物なしの場面、人型でないキャラ)ので、作者の判断で規則に合わせた。
- 権利は作者が全件確認した(`dataset/labels/rights_confirmation.json`)。
- Sonnet が `scripts/build_dataset.py` を実装し、manifest(35ケース)と画像、派生4件を組み立てた。Codex で2ラウンドのレビューを行い、収束した。DATASET_CARD を書いた。

**試行錯誤(記事用)**
- manifest のメタデータ正解で、A1111 形式の `<lora:…> fet_alisa_uniform,` のように LoRA タグとトリガーが同じカンマ区切り要素に入るケースで、トリガーを見落とすバグがあった。オーケストレータが manifest を目視して発見し、修正した。
- 修正後、メタデータ正解(独立実装)と分類器の抽出コードの結果を35件すべてで突き合わせ、一致を確認した(二重実装による検証)。
- Codex CLI は、あぐら・素足の少女の線画化を安全判定で止めたが、Antigravity は同じ依頼を処理した。

**判断**
- 正解付与の初期値を狙いにするのは作者の判断(確認・変更できる形で先入観を許容)。記録は `dataset-plan` 0.3 と DATASET_CARD に残した。

**次の一手**
- PR の順番: #4(分類コア)→ Phase 2b(評価器)→ データセット(このブランチ)。#4 は merge 待ち。
- その後 Phase 3: GPU を使う評価(E1/E1-J: Qwen、E2/E2-J: Gemma)。使う前にユーザーに確認する。

## 2026-09-26 — E1b/E2b の準備(GPU 使用前で待機)

**やったこと**
- 作者の提案で、none of the above の文言は残し、説明文(`none_criteria`: no character appears in the image)を添える方式にした。taxonomy 0.4.1(正解ラベルは不変)。other_original の説明文も「Alisa でも second_original でもない人物」に明確化した。公平性のため、JSON 方式にも「空リストは人物なしの場合だけ」の注記を入れた。
- Sonnet が実装した(テスト111件)。Codex で1ラウンドのレビューを行い、指摘なし。PR #6 の merge 後、ブランチを main に乗せ直した。

**待機理由**
- 作者が別の用途で VRAM を使うため、E1b/E2b の実行前で待機している。

**次の一手**
- 作者の合図で E1b(Qwen)→ E2b(Gemma)を手順書どおり実行する。出力は `results/E1b_qwen-lmstudio`・`results/E2b_gemma-lmstudio`、E1/E2 は残す。その後、Phase 3 の記録をまとめて PR #7 にする。

## 2026-09-26 — Phase 3: E1/E1-J(Qwen)・E2/E2-J(Gemma)の実行

**やったこと**
- 作者の許可を得て、別プロジェクトのモデルをアンロードし、LM Studio で Qwen3.5 9B → Gemma 4 12B の順にロード・評価・アンロードした。どちらも RTX 3090(GPU 1)で動作した。手順書どおり probe → M01 の1件試打(両モード)→ 全件の順で進めた。
- E1/E1-J(Qwen): 2026-09-26 11:50:48〜11:58:10 UTC(7分22秒)。E2/E2-J(Gemma): 11:58:56〜12:04:07 UTC(5分11秒)。35ケース × choice/json、ウォームアップ1回。結果は `doc/experiments/E1_qwen-lmstudio/`・`E2_gemma-lmstudio/`(run.json・cases.csv・summary.md)。
- `run.json` の `tool_commit.dirty=true` は、評価直前に実行環境 JSON へ `gpu_used` などを書き足したため(内容は run.json の runtime_info に全文が写っている。コードは `5f20596` から変更なし)。

**主な観察(小標本なので傾向として扱う。数値は summary.md が一次資料)**
- Qwen の選択式で、character の other_original が一度も付かなかった(FN 11/11)。Alisa・second_original でない人物に対して、モデルは「other character」より「none of the above」を選んでいた(例: G05 は none 0.957 / other 0.042)。設計では none を「人物なし」の意味で使っているが、選択肢の文言が「none of the above」なので「上のどれでもない」と読める。**選択肢の文言設計の問題**で、選択式の手法の限界ではない可能性が高い。Gemma では other_original の recall が 0.73 で、同じ問題は小さい。
- JSON 方式の形式不正は両モデルとも2件。単一選択の軸をリストで返していた(Qwen: O01 の situation、O06 の subject)。
- Gemma の選択式で、yes/no 確認の2件(G03・G05)がサーバー側の Channel Error(「Engine protocol predict request failed: fetch failed」)で失敗した。LM Studio のサーバーログで確認した。評価器は失敗として分母に残している。
- 処理時間(元画像31件の平均): Qwen の選択式は約10.3秒(平均11.2リクエスト)、JSON は約2.3秒。Gemma の選択式は約6.4秒、JSON は約2.5秒。
- 並行して、別プロジェクトのパッチ適用版 llama-server が常駐していた(runtime JSON に記録)。

**次の一手(作者の判断待ち)**
- 選択肢の文言を直した再実験(新しい実験IDにし、E1/E2 の結果は残す)を行うか。

## 2026-09-26 — Phase 3 の準備(GPU 使用前で待機)

**やったこと**
- PR #4(分類コア)・#5(評価器)が merge された。データセットのブランチを main に乗せ直し(worklog と README の衝突を解消)、PR #6 を出した。main 側の venv で pytest 105 passed、check-manifest OK、manifest/taxonomy の SHA256 が DATASET_CARD と一致することを確認した。
- 実行環境の記録用 JSON(`doc/experiments/runtime/{qwen,gemma}-lmstudio.json`)と実行手順書(`doc/experiments/phase3-runbook.md`)を作った。LM Studio の実行エンジンは llama.cpp CUDA12 2.41.0、CLI commit 69d945a。GPU は 4060 Ti 16GB / 3090 24GB、ドライバ 591.86。

**待機理由**
- 作者が別プロジェクトで VRAM を使用中のため、モデルのロード直前で待機している。LM Studio には別プロジェクトのモデル(qwen3.5-9b-uncensored-hauhaucs-aggressive)がロード中なので、評価のリクエストで自動ロードが起きないよう、作者の確認後に明示的にロード・アンロードする。

**次の一手**
- 作者の合図で E1/E1-J(Qwen)→ E2/E2-J(Gemma)を手順書どおり実行する(probe → 1件試打 → 全件)。

## 2026-09-25 — Phase 1: 7軸化とデータセット31枠の生成

**やったこと**
- 作者の判断で taxonomy を 0.4.0(7軸: image_type・art_style・color・subject・situation・outfit・character)に拡張した。データセット計画を 0.2 にし、服装・場面・色の群 O01〜O07 を追加した(元画像31枠)。Codex(gpt-6-luna)で5ラウンドのレビューを行い、収束した。
  - 拡張は Forge 担当10枚の生成後・正解付与前。生成記録は拡張前にコミット `7cb8882` で固定した。
- 生成スクリプト(Forge/ComfyUI・送信済みマーカー・上書き拒否・LoRA 事前確認)を Sonnet が実装した。Codex で3ラウンドのレビューを行い、収束した。
- 生成: Forge Neo 14枠、ComfyUI 10枠、外部7枠(Codex CLI 4: G06・G08・N01・C02元画像、Antigravity CLI 3: G07・N02・M03)。すべて attempt 1、作り直しなし。C02 は人工メタデータを書き込んだ(画素一致を検証)。
- 検収用の一覧ページ(artifact、非公開)を作った。各枠の画像・7軸の狙い・確認ポイントを載せている。

**試行錯誤(記事用)**
- Forge Neo は API の `override_settings` でのチェックポイント切り替えが効かなかった。最初の試し生成は `paleVeilAnima_alternative` で生成されてしまい、データセットには使っていない(`dataset/staging/_setup/`)。`/sdapi/v1/options` でモデルを切り替えて解決し、作業後に元へ戻した。
- ComfyUI の LoRA は `_comfy` 変換版のファイル名になるので、taxonomy の照合名に追加した(0.3.0)。実画像で A1111 形式・ComfyUI 形式とも照合できることを確認した。C01 は重み0でも記録として一致した。
- Codex CLI(0.155.1)は `-SandboxMode workspace-write` を指定してもワークスペースに保存できず、画像は `~/.codex/generated_images/` に出力された。依頼後に作られた画像がちょうど1枚であることを確認して回収した。
- Antigravity(agy 1.2.7)は ARTIFACT_PATH を報告させる方式で、`antigravity-artifact import` で検証してから取り込んだ。
- 1枚あたりの生成時間: Forge 約29秒、ComfyUI 約22〜74秒。

**レビュー運用の気づき**
- Codex の指摘には誤りもあった(`LoraLoaderModelOnly` に `strength_clip` が必要という指摘)。ComfyUI の `/object_info` で実際の定義を確かめ、却下が正しかったことを確認した。

**次の一手**
- 作者による検収(除外の有無)と、31枠 × 7軸の正解付与。
- 採用画像を `dataset/images/` に移し、派生ケース(A01〜A04 の strip)を作って manifest を書き、`dataset-v1.0.0` として固定する。
- PR: #4(分類コア)は merge 待ち。Phase 2b(評価器)とデータセット生成は、その後に順に PR にする。

## 2026-09-24 — Phase 2b(評価器)

**やったこと**
- Sonnet が実装した: `check-manifest`(必須項目・画像の存在・SHA256・ID の重複・派生関係の検証)、`evaluate`(権利未確認ケースの除外と派生の連動除外、ウォームアップ、ケースごとにモード順を反転する逐次実行、ケース別 JSON・`run.json`・`cases.csv`・`summary.md` の出力)。テストは77件。
- Codex(gpt-6-luna)で5ラウンドのレビューを行い、第5ラウンドで「指摘なし」になり収束した(上限ちょうど)。R1: Major 5・Minor 1(3件を却下)、R2: Major 3・Minor 2、R3: Major 3・Minor 3(2件を却下)、R4: Major 4・Minor 1(1件を却下)、R5: なし。

**判断**
- **分母**: R1 は「派生を独立に数えるな」、R2 は「全ケースを分母に」と逆向きの指摘だった。オーケストレータが確定した: 画像判定の主集計は元画像を分母にし(水増し防止)、全ケース分母の値も併記する。メタデータの集計は全ケースを分母にする。計画書 §5 に追記した。
- 実行環境(GGUF/mmproj の SHA256、サーバー種別・commit・パッチ有無・起動引数・GPU)はサーバーから自動取得できないので、人が書いた JSON を `--runtime-info` で `run.json` にそのまま写す。
- データセット版は `--dataset-version` で記録する。DATASET_CARD の manifest/taxonomy SHA256 と突き合わせる運用にした(dataset-plan §4 を更新)。
- メタデータの採点は format・characters・loras(名前+重みの多重集合)・trigger_words。`artificial` は検出できないので採点対象外。

**気づき(レビュー運用)**
- Codex は各ラウンドで新しい細部を出し続ける傾向があり、R2 では R1 と逆の要求も出た。依頼文に「確定事項(再提起不要)」を列挙すると収束が早まった。

**次の一手**
- PR #4 の merge 後に PR #5 を出す。
- 画像生成の準備(GPU を使う前にユーザーに確認する)。

## 2026-09-24 — Phase 2a(最小の分類コア)

**やったこと**
- Sonnet が実装した: taxonomy 読み込み、LoRA/トリガーワード抽出(A1111/Forge と ComfyUI の LoraLoader)、画像前処理、OpenAI 互換バックエンド、logprobs 集計と選択式判定(複数キャラはランキング+yes/no)、通常 JSON ベースライン、1枚分類パイプライン、CLI(`probe`・`classify`)。テストは38件。
- Codex(gpt-6-luna)で3ラウンドのレビューを行い、収束した。R1: Major 4・Minor 1(時間計測の1件は §5.1 と逆なので却下)と、オーケストレータの指摘4件。R2: Major 2・Minor 1(一部反映)。R3: Major 1・Minor 3(すべて理由付きで却下)。
- 主なバグ修正: 人物なし画像に other_original が付く、JSON の回答例が実在の選択肢(alisa 等)で誘導していた(公平性)、確認失敗が隠れる、通信失敗で全体が停止する。
- 実サーバーでのスモーク(LM Studio + Qwen3.5 9B、非公開の Alisa 学習画像1枚。評価値ではない): probe は成功。choice は5リクエストで 5,141 ms、json は1リクエストで 1,708 ms。両方式とも alisa と判定。
  - 旧 SDXL 用 LoRA 名 `fet_alisa_uniform` は taxonomy に列挙していないので、メタデータ一致なし(設計どおり)。
  - choice は軸ごとに画像を再送するので、1軸あたり約0.9〜1.3秒かかる。キャッシュ改造(E3/E4)の効果を測る意味がありそう。

**判断**
- 確認(yes/no)が1件でも失敗した multi 軸は、軸ごと失敗として扱い、catch_all に逃がさない(失敗を隠さない)。
- taxonomy を 0.2.0 にした(second_original の criteria を計画の衣装に合わせた)。

**ユーザー指定**
- VRAM は複数プロジェクトで取り合いなので、GPU を使い始める直前に必ずユーザーを呼ぶ(CLAUDE.md に追記)。

**次の一手**
- Phase 2b: manifest を読んで全件評価し、ケース別 CSV と集計 Markdown を出す(GPU 不要)。
- 画像生成の準備(GPU を使う前にユーザーに確認する)。

## 2026-09-24 — Phase 1 計画(正解付与規則・生成計画表)

**やったこと**
- `doc/dataset-plan.md` を作成した(画像生成前に固定)。内容は、正解付与規則(4軸・メタデータ正解・other/空集合の採点)、24元画像+派生4の生成計画表、manifest 仕様、検収・版固定の手順。
- Codex(gpt-6-luna)で5ラウンドのレビューを行い、第5ラウンドで「指摘なし」になり収束した。指摘の数は R1: Critical 2・Major 5・Minor 2、R2: Major 2、R3: Major 6・Minor 3、R4: Major 2、R5: なし。
- `subject` を単一ラベルに確定した(basic-design §4 も更新)。

**判断**
- character は「必須特徴/見えていれば合う特徴」の表で判定し、一意に決められない人物がいる画像は除外する。似たキャラ群が測るのは「服装の手掛かりがある状態」まで(服装が隠れた識別は初版の対象外)。
- 後選びを防ぐため、1枠1枚・作り直しは最大2回、落ちたら欠数として公表し補わない。狙いと違うラベルになった画像は、画像どおりのラベルで残す。
- C01(LoRA 重み0)は本物の矛盾ケース、C02 は人工的に書き込んだメタデータであることを明記する。
- R4 の指摘1件の前半は誤読と判断して却下した(衣装の核は各キャラの必須特徴なので一意に決まる)。抜けていた「どちらの衣装でもない」ケースだけを補った。

**未解決・次の一手**
- taxonomy/default.yaml の second_original の criteria を計画の衣装(紺ブレザー・緑ネクタイ・グレーのプリーツスカート)に合わせる → Phase 2a の PR で対応する。
- 画像生成の準備: Forge Neo/reForge と ComfyUI の起動はユーザーに依頼する。外部生成(Codex/Antigravity)の手段を確認する。

## 2026-09-24 — Phase 0(出典・モデル・環境・利用条件)

**やったこと**
- 移植元 `自家製の画像分類アプリ(非公開)`@`7994960` の分類コアを Sonnet で要約した。丸ごとは移植せず、要点だけを最小実装で引き継ぐ方針にした(`doc/phase0-sources-models.md` §1)。
- 通常版の Qwen3.5 9B / Gemma 4 12B(lmstudio-community、Q4_K_M + BF16 mmproj)をダウンロードし、SHA256 を記録した。
- LM Studio で probe した。両モデルとも `reasoning_effort: "none"` があれば、画像入力・1文字回答・logprobs が成立する。無いと thinking で失敗する。
- 画像生成は Anima base v1.0 を使う。生成物の公開は可能(ライセンス §2.e)。Alisa の Anima 用 LoRA は `fet-alisa-uniform-anima-v4u`。
- llama.cpp の改造版の所在を確認した(上流 `f95b0d9` + 1行パッチ)。未改造版は同じコミットを別クローンでビルドする(Phase 4)。

**判断**
- LoRA 照合は「列挙した名前集合との完全一致」に変更する(LoRA の版ごとに名前が違うため)。
- 全件 SFW(ユーザー指定)。似た別キャラは LoRA なしのプロンプト設計(ユーザー指定)。

- 似た別キャラは、顔・髪を Alisa に似せ(茶髪ボブ・青目)、服装を変える(ユーザー指定)。
- LoRA 付き画像は Forge Neo/reForge と ComfyUI の両方で生成する(ユーザー指定)。抽出は A1111 形式と ComfyUI の LoraLoader の2経路。
- note の実験記録記事にする前提なので、試行錯誤も含めてその都度記録する(ユーザー指定、AGENTS.md に追記)。

**未解決・次の一手**
- Alisa の容姿基準を確定(`doc/phase0-sources-models.md` §4)。タイツは薄く肌色に見えるケースがあるため判定に使わない(ユーザー指定)。
- LoRA もトリガーワードも無い画像では、根拠は容姿だけになる(想定どおり。ユーザーと確認)。
- Phase 1: 正解付与規則と24元画像の生成計画表を作る。
- 並行して Phase 2 の分類コアの実装(Sonnet)に入れる。

**関連PR**: Phase 0 記録の PR
