# データセット計画 — 正解付与規則・生成計画表・manifest 仕様

- 文書版: 0.1(画像生成前に固定)
- 作成日: 2026-09-24
- 対応: `implementation-experiment-plan.md` Phase 1、`requirements.md` §4
- **この文書は画像を1枚も生成していない時点でコミットする**。生成後に計画を変えた場合は、予定と実績を別列に残し、変更理由を書く(都合のよい画像の後選びを防ぐため)。

## 1. 全体方針

- 元画像24枚。全件SFW(Gemma も全件で評価する)。
- 分類体系は `taxonomy/default.yaml`(4軸: `image_type`、`art_style`、`subject`、`character`)。**taxonomy の候補と criteria は画像生成前に確定し、版と SHA256 を固定する**。生成後に基準を変える場合は、全画像の正解を新基準で付け直し、別のデータセット版として扱う。
- 似た別キャラ群が測るのは「服装の手掛かりがある状態で、顔・髪が似たキャラを Alisa と取り違えないか」まで。服装が隠れた状態(顔のアップ等)での識別は初版の対象外。
- 評価はすべて `split: test`。しきい値は元実装の値(yes/no の採用 0.5、候補フロア 0.001)を固定で使い、このデータセットで調整しない。調整が必要になったら新しい実験IDを振る。
- 派生ケース(メタデータ除去コピー)は元画像と同じ `source_image_id` で束ね、元画像数に数えない。

## 2. 正解付与規則

正解は**画像を目視して付ける**。生成時の狙い(計画表の「狙い」列)は正解ではない。モデルの出力を見る前に付けて固定する。

### 2.1 `image_type`(単一)
- `comic`: コマ割り、または吹き出しがある。
- `ui`: アプリ・ゲーム・Web の画面、またはそのモックアップ。
- `illustration`: 上記以外の1枚絵。
- `other`: どれにも当たらない(写真風の画像など)。

### 2.2 `art_style`(単一)
- `anime_2d`: セル調の2Dアニメ・漫画絵。
- `painterly_2d`: 筆致や柔らかい塗りが見える2D絵画調。
- `pixel_art`: 四角いピクセルが見える低解像度のドット絵。
- `other`: 3DCG調、写実調、フラットなUIデザインなど。
- アニメ絵で塗りが厚い場合など迷うときは、線画と平塗りが主なら `anime_2d`、塗りの筆致が主なら `painterly_2d`。

### 2.3 `subject`(単一、主題を1つ)
- 単一ラベルとする(集合にしない)。
- 人物・人型キャラが画面の1割以上を占めるか、視線の中心にいれば `person`(ほかに目立つ主題があっても `person`)。
- そうでなければ、人物以外の主題のうち画面上の面積が最も大きいもの(`landscape` 等)。
- `image_type` が `ui` の画像は、画面の内容にかかわらず常に `other`。

### 2.4 `character`(集合、0人以上)
人物ごとに、見えている特徴で次の順に判定する。

| キャラ | 必須(見えて合うこと) | 見えていれば合うこと(見えなければ問わない) |
|---|---|---|
| `alisa` | 茶髪のストレートなボブ・前髪、ピンクのベスト、赤い紐リボン | 青い目、白い襟付き長袖シャツ、濃いピンクのタイトスカート |
| `second_original` | 茶髪のストレートなボブ・前髪、紺のブレザー、緑のネクタイ | 青い目、グレーのプリーツスカート |

- 必須特徴がすべて見えて合い、見えている任意特徴に基準と違うものが無ければ、そのキャラとする。**タイツは判定に使わない**。
- 見えている特徴が基準と違う(例: 目が赤い、リボンが青い)なら、そのキャラには含めない。
- 各キャラの衣装の核はそれぞれの必須特徴なので、Alisa の核(ピンクのベスト+赤い紐リボン)が見えれば second_original にはならず、その逆も同じ。
- 髪は2人の基準に合うが、どちらの衣装の核でもない服がはっきり見える人物は `other_original`。衣装の核が隠れていて服装を確認できない人物は除外対象。
- `other_original`: 人物・人型キャラで、上の2人のどちらでもないと判断できるもの(髪型・髪色が明らかに違う、など)。服装だけ Alisa や second_original と同じで髪が違う場合もここ。
- **除外**: 必須特徴が見えない(顔のアップ、衣装の核が隠れる構図など)ため、上の2人か `other_original` かを一意に決められない人物が1人でもいる画像。
- 人物がいなければ空集合。二人以上いれば、識別できる人物をすべて入れる。背景の小さな人影で識別できないものは入れない。
- LoRA の指定やメタデータは見ない。画像だけで判断する。

### 2.5 メタデータの正解(画像の正解とは別に採点する)
生成記録(実際に画像に書き込まれたメタデータ)から付ける。

- `format`: **画像に埋め込まれたメタデータの形式**(生成ツール名ではない)。`a1111`(`parameters` テキスト。Forge Neo / reForge もこの形式で書く想定で、最初の生成時に実物で確認する)/ `comfyui`(`prompt` の workflow JSON)/ `none`。生成ツール名は `generation.tool` にだけ書く。
- `loras`: 記録されている LoRA の `{name, weight}` のリスト。
- `trigger_words`: プロンプトに含まれる、taxonomy に列挙したトリガーワードのリスト。
- `characters`: `loras` の名前または `trigger_words` が taxonomy の列挙と完全一致するキャラの集合。**LoRA の重みが0でも「記録はある」ので含める**(C01)。
- `artificial`: メタデータを評価用に人工的に書き込んだ・改変した場合 true(C02)。
- メタデータの期待値(生成記録から確定する): C01 は `format: a1111`、`loras: [{fet-alisa-uniform-anima-v4u, 0}]`、`trigger_words: []`、`characters: [alisa]`、`artificial: false`。C02 は `format: a1111`、`characters: [alisa]`、`artificial: true`。
- `expected_metadata` はどのケースも、**完成画像に実際に埋め込まれている値**から付ける(C02 は書き込み後の画像から)。C02 の `generation.tool` は元画像を作った外部サービスのままにする。
- 画像の正解は C01・C02 とも §2.4 で画像から付ける(狙いは C01 が `second_original`、C02 が空集合)。
- 派生ケース(`*-strip`)の `expected_metadata` は、**除去後の画像から実際に検出されるべき値**(`format: none`、`loras`・`trigger_words`・`characters` は空、`artificial: false`)にする。元画像の生成記録は `generation`(元画像のもの)で参照する。

### 2.6 採点上の `other`・空集合・`none` の扱い
- 単一軸(`image_type`、`art_style`、`subject`)の `other` は通常のラベルとして扱い、完全一致で採点する。
- `character` は集合で採点する。モデルが `none of the above` を選び、確認でも誰も採用されなかった場合は空集合として扱う。空集合どうしは完全一致とみなす。キャラごとの TP/FP/FN には空集合は寄与しない。
- 推論失敗・形式不正のケースは不正解として分母に残す(別途、失敗件数も出す)。

### 2.7 除外と生成枠の運用(後選びの防止)
- 各枠は計画した seed で**1枚だけ**生成し、検収に通ればそれを採用する。同じ設定で複数枚を作って良いものを選ぶことはしない。
- 検収で落ちた場合だけ、seed を1つ進めて作り直す(1枠あたり生成は最大2回。2回とも落ちたら、その枠は欠数として公表し、別の手段や追加生成で補わない)。作り直しは理由とともに `dataset/generation/log.csv` に残す。
- 除外理由(全枠共通の唯一の基準): 2.4 で一意に判断できない人物が1人でもいる、画像が破綻している、ほかの画像と構図・内容がほぼ同じ(重複)、権利が不明、SFW でない。
- 狙いと違うラベルになった画像(例: N01 が Alisa に見えない)は、除外せず**画像どおりのラベルで残す**。その結果、群ごとの件数が計画(要件 §4.1 の目安)を下回ったら、不足として件数を公表する(埋め合わせの追加は新しいデータセット版で行う)。

## 3. 生成計画表(予定)

凡例 — キャラの生成指定: Alisa = 茶髪のストレートなボブ・前髪・青い目、白い襟付き長袖シャツ・ピンクのベスト・赤い紐リボン・濃いピンクのタイトスカート。**second_original = 茶髪のストレートなボブ・前髪・青い目、紺のブレザー・緑のネクタイ・グレーのプリーツスカート**(どちらもプロンプトに明記する。Alisa は LoRA 群ではトリガーワードも入れる)。**Alisa・second_original を描く枠は、上半身の衣装の核が見える構図(上半身以上が写る)に固定する**。生成: `Forge` = Forge Neo / reForge、`Comfy` = ComfyUI、どちらも Anima base v1.0。`外部` = Codex / Antigravity 等の画像生成(サービスは生成時に記録)。LoRA = `fet-alisa-uniform-anima-v4u`。seed は生成時に固定して記録する。

| ID | 群 | 狙い(正解ではない) | 生成 | LoRA/メタデータ |
|---|---|---|---|---|
| G01 | 一般 | 夕暮れのファンタジーの街並み、人物なし / illustration・anime_2d・landscape | Forge | なし |
| G02 | 一般 | 格納庫に立つ巨大ロボット、人物なし / illustration・anime_2d・mecha_vehicle | Comfy | なし |
| G03 | 一般 | 果物とティーポットの静物 / illustration・painterly_2d・object | Forge | なし |
| G04 | 一般 | 森の白い狐の精霊、人物なし / illustration・anime_2d・creature | Comfy | なし |
| G05 | 一般 | 港の老漁師 / illustration・painterly_2d・person・other_original | Forge | なし |
| G06 | 一般 | 丘の上の城のドット絵 / illustration・pixel_art・landscape | 外部 | なし |
| G07 | 一般 | 4コマ漫画のページ(オリジナル人物) / comic・anime_2d・person・other_original | 外部 | なし |
| G08 | 一般 | ゲームのインベントリ画面のモックアップ / image_type: ui、art_style: other、subject: other | 外部 | なし |
| A01 | Alisa LoRAあり | オフィスで腰から上、正面 | Forge | LoRA 1.0 + トリガー |
| A02 | Alisa LoRAあり | 街角で全身、横向き | Forge | LoRA 0.8 + トリガー |
| A03 | Alisa LoRAあり | カフェで上半身 | Comfy | LoRA 1.0 + トリガー |
| A04 | Alisa LoRAあり | 公園のベンチに座る | Comfy | LoRA 0.8 + トリガー |
| N01 | Alisa LoRAなし | Alisa の参照画像(作者の既存画像)を与えて、同じ容姿・衣装で別の場面を描かせる。参照画像はデータセットに含めない | 外部 | なし |
| N02 | Alisa LoRAなし | 同上(N01 と別のサービス) | 外部 | なし |
| N03 | Alisa LoRAなし | 容姿基準をプロンプトだけで記述 | Forge | なし(LoRA もトリガーも入れない) |
| S01 | 似た別キャラ | second_original、教室で上半身 | Forge | なし |
| S02 | 似た別キャラ | second_original、駅のホームで全身 | Comfy | なし |
| S03 | 似た別キャラ | second_original、図書館で座る | Forge | なし |
| S04 | 似た別キャラ | second_original、屋上で横向き | Comfy | なし |
| M01 | 二人以上 | Alisa と second_original が並ぶ | Forge | LoRA 0.8 + トリガー |
| M02 | 二人以上 | Alisa と other_original(男性の同僚) | Comfy | LoRA 0.8 + トリガー |
| M03 | 二人以上 | second_original と other_original | 外部 | なし |
| C01 | メタデータと画像の矛盾 | second_original の生成指定で描かせ、LoRA を**重み0**で指定(本物の生成記録。記録はあるが効いていない)。トリガーワードは入れない | Forge | LoRA 0、トリガーなし |
| C02 | メタデータと画像の矛盾 | 人物なしの風景画像に、Alisa の LoRA を指定した A1111 形式の生成情報を**人工的に書き込む**(評価用に作った改変メタデータと明記)。書き込んだテキスト全文と書き込み手順(スクリプト)を `generation.params_file` に残す | 外部 → 書き込み | 人工 |

### 派生ケース(元画像数に数えない)

| ID | 元 | 内容 |
|---|---|---|
| A01-strip〜A04-strip | A01〜A04 | メタデータ(テキストチャンク)を除去したコピー。画素は同一 |

合計: 元画像24 + 派生4 = 28ケース。

### 計画上の注意
- **検収と正解は、どの枠も §2.4・§2.7 の規則だけで決める**(狙いに合うかどうかでは採否を決めない)。§2.4 で判断できない人物がいれば除外して作り直し、判断できれば狙いと違っても画像どおりのラベルで残す。
- M01 は LoRA が画面全体に効くため、second_original まで Alisa の衣装・髪に寄る恐れがある。寄って特徴が混ざれば §2.4 で除外・作り直し(2回とも落ちれば欠数)。
- N03 は LoRA なしでは Alisa に見えない可能性が高い。見えなければ画像どおりのラベルで残す。
- 矛盾群(C01・C02)は、画像の正解が `alisa` を含まない限り「メタデータは Alisa、画像は Alisa でない」矛盾ケースとして成立する。C01 の画像が `alisa` と判定された場合も除外せず残し、矛盾群の件数不足として公表する。一意に判断できない人物がいる場合は、ほかの枠と同じく除外する。
- C02 の書き込みは、元画像の画素を変えずにテキストチャンクだけを加える。

## 4. manifest 仕様(`dataset/manifest.jsonl`、1行1ケース)

```json
{
  "case_id": "A01",
  "source_image_id": "A01",
  "derived_from": null,
  "image_path": "dataset/images/A01.png",
  "image_sha256": "<sha256>",
  "split": "test",
  "scenario": "alisa_lora",
  "expected": {"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]},
  "expected_metadata": {"format": "a1111", "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0}], "trigger_words": ["fet_alisa_uniform"], "characters": ["alisa"], "artificial": false},
  "generation": {"tool": "forge_neo", "tool_version": "<版/commit>", "base_model": "anima-base-v1.0", "base_model_sha256": "bd43b7cf...", "seed": 123, "attempt": 1, "params_file": "dataset/generation/A01.json"},
  "rights": {"terms": "CircleStone Labs Non-Commercial License v1.2 §2.e (Outputs)", "rights_confirmed": true},
  "review": {"status": "accepted", "note": null}
}
```

- `scenario`: `general` / `alisa_lora` / `alisa_no_lora` / `similar` / `multi` / `conflict`。派生ケースは元と同じ scenario に `derived_from` を付ける。
- `generation.tool`: `forge_neo` / `reforge` / `comfyui` / `external:<サービス名>`。`attempt` は 2.7 の作り直し回数(1始まり)。
- `generation.params_file`: 生成に使ったプロンプト・パラメータ・ワークフローの全文。外部サービスは、実際に渡した依頼文・参照画像の有無・サービス名と利用日を書く。公開前に個人情報や固有作品名が無いか点検する。
- `rights.terms`: 生成物の利用条件の根拠(ライセンス名と条項、外部サービスは規約名と確認日)。確認は作者が行う。
- 除外・作り直しの履歴は manifest ではなく `dataset/generation/log.csv`(枠ID、attempt、seed、結果、理由)に残す。
- 評価器は manifest にない画像を拾わない。`rights_confirmed` が true でないケースは公開用の結果に含めない。
- データセット固定時に、データセット版・manifest の SHA256・taxonomy のバージョンと SHA256 を `dataset/DATASET_CARD.md` に記録する。評価の実行記録(`run.json`)は実際に使った manifest と taxonomy の SHA256 を記録し、DATASET_CARD の値と突き合わせて版を確認する。

## 5. 検収と版の固定

1. 生成した画像は `dataset/staging/`(Git 管理外)に置く。
2. 作者が目視で検収し、採用・除外(理由付き)を決め、採用画像に正解を付ける。
3. 採用画像を `dataset/images/` に移し、派生ケースを作り、manifest を書く。
4. manifest の整合(参照画像の存在、SHA256、派生関係、件数)を確認して `dataset-v1.0.0` として固定する。評価中は編集しない。
