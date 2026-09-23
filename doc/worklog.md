# 作業記録

新しい順。compact 後の文脈復元用。詳細は各 PR と `doc/` を参照。

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

**未解決・次の一手**
- ユーザー確認: Alisa の容姿基準(学習画像からの下書きを提示)、似た別キャラの設計、Anima で LoRA 付き画像を作るツール(ComfyUI なら LoraLoader の抽出が必要)。
- Phase 1: 正解付与規則と24元画像の生成計画表を作る。
- 並行して Phase 2 の分類コアの実装(Sonnet)に入れる。

**関連PR**: Phase 0 記録の PR
