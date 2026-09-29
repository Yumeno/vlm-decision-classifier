# DATASET_CARD — vlm-decision-classifier 評価用データセット

| 項目 | 値 |
|---|---|
| データセット版 | **v1.0.0** |
| manifest | `dataset/manifest.jsonl`(35行)SHA256 `941faf84f0c1e3f027748766b6ce2ec4181cc8749b0c1ad70ec8113b528baab0` |
| taxonomy | `taxonomy/default.yaml` 0.4.0、SHA256 `ec7b2a2f750c7848c56033c9b4a5aee799df1aa7548c126528e6cfc1db4b26aa` |
| taxonomy を固定した日 | 2026-09-25(0.4.0。正解付与より前) |
| 正解付与を始めた日 | 2026-09-25(正解付与シートで作者が31枠を確定) |
| 計画 | `doc/dataset-plan.md` 0.3(正解付与規則・生成計画・例外の経緯) |
| 組み立て | `scripts/build_dataset.py`(ラベル・生成記録・画像から manifest を生成し、SHA256・メタデータ・派生の画素一致を検証) |

評価の実行記録(`run.json`)に残る manifest と taxonomy の SHA256 を、この表の値と突き合わせて版を確認する。

## 構成

- 元画像31枚 + 派生4件(`A01-strip`〜`A04-strip`、A01〜A04 のメタデータ除去コピー。画素は同一)= 35ケース。すべて `split: test`、全件 SFW。
- 群: general 8 / alisa_lora 4(+派生4) / alisa_no_lora 3 / similar 4 / multi 3 / conflict 2 / outfit_scene 7。
- 生成手段(元画像): Forge Neo 13、ComfyUI 6、Codex CLI 6、Antigravity CLI 6。Forge Neo / ComfyUI は Anima base v1.0(`anima-base-v1.0`、SHA256 `bd43b7cf…a006e`)。Alisa の LoRA は作者の自作 `fet-alisa-uniform-anima-v4u`(ComfyUI 用は `_comfy` 変換版)。LoRA 本体は同梱しない。
- 画像は生成物のバイト列をそのままコピーしている(再エンコードしていない)。Forge Neo / ComfyUI の画像には生成時のメタデータ(A1111 形式の `parameters` / ComfyUI の `prompt`)が入っている。

## 正解の分布(元画像31枚)

| 軸 | 分布 |
|---|---|
| image_type | illustration 29 / comic 1 / ui 1 |
| art_style | anime_2d 26 / painterly_2d 3 / pixel_art 1 / other 1 |
| color | full_color 27 / grayscale 2 / line_art 2 |
| subject | person 24 / landscape 3 / mecha_vehicle 1 / object 1 / creature 1 / other 1 |
| situation | daily 16 / other 7 / work_study 3 / meal 2 / sports 1 / battle 1 / hobby 1 |
| outfit(集合) | office_wear 9 / school_uniform 7 / other 5 / sailor_uniform・maid・miko・nun・swimsuit 各1 / 空 7 |
| character(集合) | other_original 11 / alisa 9 / second_original 7 / 空 7 |

メタデータ(35ケース): a1111 14 / comfyui 6 / none 15。Alisa の LoRA またはトリガーが記録されているのは A01〜A04・C01・C02・M01・M02 の8枠(C01 は LoRA 重み0、C02 は評価用に人工的に書き込んだメタデータで `artificial: true`)。

**偏りと限界**: 正解は小標本で、単一軸の多くは1〜3件しかない候補がある(comic・ui・pixel_art・grayscale・line_art など)。候補別の成績は個別例として読み、強い結論にしない。

## 正解の付け方と記録

- 作者が画像を見て、`doc/dataset-plan.md` §2 の規則で付けた。生成時の狙いを初期値として表示し、作者が確認・変更して確定する方式(作者の判断)。
- 一次記録: `dataset/labels/labels_initial_readback.json`(確定直後の読み出し)と `dataset/labels/labels_final.json`(最終)。確定後に5枠(G01・G02・G03・G04・G06)を、規則(人物なしの場面は other、キャラは人物・人型キャラのみ)に合わせて作者承認のもと修正した。
- メタデータの正解(`expected_metadata`)は、完成画像に実際に埋め込まれている値から、分類器とは独立した読み取りで付けた。分類器の抽出コードの結果と35件すべてで一致することを確認した。

## 生成の経緯(要約)

- 1枠1枚の運用。作者の検収で6枠(G02・G07・O02・O04・O06・O07)の1回目を除外し、作者の指示で外部ツールによる編集または新規生成で2回目を作った(dataset-plan 0.3 の例外)。O07 は Codex の安全判定で出力されず、Antigravity で同じ依頼を行った(attempt `2b`)。
- 依頼文、生成ツールに渡されたプロンプト、SHA256、試行ごとの結果は `dataset/generation/`(`log.csv`、各 `{枠}_a{回}.json`、送信済みマーカー)に残している。

## 権利と利用条件

- 31枚すべて、作者が公開可能であることを確認済み(`dataset/labels/rights_confirmation.json`、2026-09-25)。
- Anima(Forge Neo / ComfyUI)の生成物: CircleStone Labs Non-Commercial License v1.2 §2.e により、生成物は用途を問わず利用できる(モデルと LoRA は非商用。どちらも同梱しない)。
- Codex CLI の生成物: OpenAI の規約上、出力は利用者のもの。Antigravity CLI の生成物: Google の規約上、出力は利用者が利用できる(いずれも 2026-09-25 確認)。
- N01・N02 は、作者の自作キャラクター Alisa の既存画像を参照画像として使った(参照画像はデータセットに含めない)。
- Alisa は作者のオリジナルキャラクター。

## ライセンス(画像)

著作権者: 資材部の懲りない面々(公式: http://jmd.ickx.jp/fet 、2026-09-29 作者決定)

| 対象 | ライセンス |
|---|---|
| 正解の character に Alisa を含む画像: A01〜A04、M01、M02、N01〜N03(元画像9枚)と派生 A01-strip〜A04-strip | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| それ以外の元画像22枚(第2のオリジナルキャラクターだけが写る画像、メタデータにだけ Alisa の記録がある C02 を含む) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

- 対象は `dataset/images/` の画像ファイル。manifest・正解ラベル・生成記録などのテキストの扱いは、コードのライセンス確定時に明記する(未確定)。
- クレジット表記の例: 「画像: 資材部の懲りない面々 http://jmd.ickx.jp/fet (vlm-decision-classifier データセット、CC BY-NC-SA 4.0 / CC BY 4.0)」。
