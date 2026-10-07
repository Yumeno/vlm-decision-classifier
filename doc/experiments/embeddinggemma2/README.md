# EmbeddingGemma 2 のゼロショット分類の試し

2026-10-07 の記録。**各1回、元画像31枚の小標本**なので強い結論は出さない。スコアはコサイン類似度で、確率ではない。数値の出典は、このフォルダの `probe_output.txt`(`scripts/embeddinggemma_probe.py` の実際の出力)。

## 1. 目的

`google/embeddinggemma-2`(画像とテキストを同じ768次元空間に埋め込むモデル、740M、Apache-2.0)を、生成せずに**コサイン類似度だけ**で選択式分類に使えるかを試す。

## 2. 条件

- ランタイム: 未改造(vanilla)の llama.cpp b11461(commit `4d756bc`)。EmbeddingGemma 2 への対応を含む版で、推奨版 b11447 には未対応のため。
- 重み: `ggml-org/embeddinggemma-2-GGUF` rev `bfcd298` の `embeddinggemma-2-BF16.gguf` と `mmproj-embeddinggemma-2-BF16.gguf`(**重みは本リポジトリに含めない**)。
- 起動(RTX 3090 のみ。`CUDA_VISIBLE_DEVICES` で 3090 だけを見せた):

```
llama-server -m <モデルのディレクトリ>/embeddinggemma-2-BF16.gguf --mmproj <モデルのディレクトリ>/mmproj-embeddinggemma-2-BF16.gguf --embeddings -ngl 99 --host 127.0.0.1 --port 1235
```

- データ: 元画像31枚、`taxonomy/default.yaml`、各1回。

### 事前に固定した設計(流す前に決め、31枚を見て変えていない)

- 選択肢の文: `task: classification | query: <name>: <criteria>`(分類体系の名前と説明文をそのまま。評価後に書き換えていない)。
- 画像: 長辺1024 の JPEG、前置きなし。
- 単一選択の項目は、画像と各選択肢の文のコサイン類似度の argmax。
- 複数選択(服装・キャラ)は採点せず、類似度の上位3を見るだけ。
- `/v1/embeddings` の入力は、各要素を `{"content": [...]}` で包む形式。

## 3. 結果(元画像31枚、各1回)

| 項目 | 正答 |
|---|---|
| 画像の種類 | 23/31 |
| 画風 | 27/31 |
| 色 | 7/31 |
| 被写体 | 11/31 |
| 状況 | 24/31 |
| **合計** | **92/155** |

比較の参考(別方式の記録からの引用):

- DiffusionGemma(`criteria`、画像トークン 140、seed 0): 単一5項目 143/155、1枚 453 ms([`../dgemma/followup-12.md`](../dgemma/followup-12.md) §6.3)。
- OpenJev: 単一5項目 144/155(default・criteria とも)([`../openjev/README.md`](../openjev/README.md) §3)。

時間(クライアント側の HTTP 往復。`classification_wall_ms` とは境界が違う参考値):

- 画像1枚の埋め込み: 平均 113 ms(最初の1枚 434 ms)。
- 選択肢の文36本の埋め込み: 合計 770 ms(最初に1回だけ)。

## 4. 観察

- 色と被写体の正答が特に少なかった(7/31、11/31)。
- 複数選択(参考、採点なし)は、出力に載せた先頭8枚(16行)だけを見た。そのうち正解のある画像では、正解が1位に来ていた(例: A01 の服装 office_wear 0.681、キャラ alisa 0.777)。ただし alisa と second_original の差が小さい(A01: 0.777 対 0.743、A04: 0.786 対 0.761)。残りの23枚は見ていない。
- 正解なしの画像(C02、G01、G02)でも全候補が 0.5〜0.6 程度に並ぶ(例: G01 の服装 maid 0.605、キャラ second_original 0.635)。「なし」の閾値を決めにくい。

## 5. 結論(控えめに)

- 今回の条件(分類体系の説明文をそのまま使った選択肢の文、コサイン argmax)では、同じ31枚・同じ分類体系で測った DiffusionGemma・OpenJev(143〜144/155)より正答がかなり少なかった(92/155)。この条件のままでは、判定の主役には向かないと見ている。
- 速いので、前段のふるい分けや類似検索に使える可能性はあるが、未検証(推測)。
- 選択肢の文をこの方式向けに書き換えれば上がる可能性もあるが、未試行。
- 小標本・各1回なので強い結論は避ける。

## 6. 再現

llama-server を上の条件で起動してから、

```
.venv\Scripts\python.exe scripts\embeddinggemma_probe.py --url http://127.0.0.1:1235/v1/embeddings
```

出力が `probe_output.txt`。
