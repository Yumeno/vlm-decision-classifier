# 保守メモ(リポジトリの運用)

利用者向けではなく、このリポジトリを保守するときの手順と設定の記録。

## ダッシュボード(GitHub Pages)の更新

- Actions は普段は停止している。`site/` を更新して公開し直すときだけ、リポジトリ設定で Actions を一時的に有効にし、`Deploy dashboard to GitHub Pages` を手動実行(または main への push)して、終わったら再び無効にする。
- 自動更新(Dependabot の version updates)は使わない。外から新しい版が入る経路を持たないため。
- workflow の action はフル SHA で固定している(`.github/workflows/pages.yml`、バージョンは行末コメント)。更新するときは、タグが指すコミット SHA を GitHub で確かめてから書き換える。

## リポジトリ設定(公開時に適用済み)

- Wiki・Projects・Discussions 無効、Issues 有効。squash / merge commit のみ、merge 後にブランチ自動削除。
- secret scanning と push protection、Dependabot の脆弱性アラート(通知のみ)、private vulnerability reporting を有効。
- Actions(有効にしている間): 既定の `GITHUB_TOKEN` は read-only、許可する action は GitHub 製のみかつ SHA 固定必須、外部コントリビューターのワークフロー実行は承認制、Actions による PR の作成・承認は不可、`github-pages` 環境のデプロイ元は `main` のみ。
- デフォルトブランチの ruleset `protect-main`: 削除禁止・force push 禁止・PR 必須(承認数0)。
