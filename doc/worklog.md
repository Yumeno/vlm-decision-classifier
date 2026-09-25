# 作業記録

新しい順。compact 後の文脈復元用。詳細は各 PR と `doc/` を参照。

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
