# CLAUDE.md

プロジェクト共通の規約・不変条件は AGENTS.md にまとめてある(Codex と共有)。以下はそれを読み込んだ上での Claude Code 固有の補足。

@AGENTS.md

## Claude Code 固有

- **体制**: メインループは計画・要件整理・設計判断・レビュー指揮・統合に徹し、まとまった実装コーディングはサブエージェント(Sonnet 等)に委任する。委任時は該当する `doc/` の節番号と AGENTS.md の不変条件を明示して渡す。
- **レビュー**: PR前に `/ask-codex-with-context レビュー` でセカンドオピニオンを取り、Major/Critical が残る限り修正→再レビューを回す。
- **シェル**: PowerShell 主体。許可設定はプレフィックスマッチなので、コマンドは `;` / `&&` で連結せず1ツール呼び出し1コマンドにする。`.ps1` は UTF-8 BOM 付きで保存。日本語出力が化ける場合は `PYTHONIOENCODING=utf-8`。
- **venv**: `py -3.12 -m venv .venv` → `.venv\Scripts\python.exe` を明示して使う。
- **LM Studio を叩く前に**ロード中モデルを確認する(未ロードモデルへの probe が auto-load で既存モデルを押し出すことがある)。bulk 失敗時はクライアントを直す前にサーバーログを確認。
- **GPU**: RTX 4060 Ti 16GB + RTX 3090 24GB。速度比較ではどちらのGPU・オフロード設定で測ったかを必ず `run.json` に残す。
- **X/Twitter の URL は WebFetch しない**(ログイン必須)。着想元投稿の内容は `doc/requirements.md` の記録と二次情報で扱う。
