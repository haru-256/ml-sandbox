# ml-sandbox AGENTS

## 1. System Context

- このリポジトリは、機械学習モデル実装と実験を行うための Python モノレポです。
- 主な project / app は次の通りです。
    - `projects/recsys-ranking`: 推薦システムの Ranking 段階（DeepFM, DLRM, DIN, DCNv2 など）
    - `projects/recsys-candidate-generation`: 推薦システムの Candidate Generation 段階（TwoTower, SASRec, gSASRec, SimpleX, LightGCN など）
    - `projects/sentiment_analysis`: IMDB レビューを用いた Transformer Encoder のフルスクラッチ実装による感情分類。`libs/ml_sandbox_libs` に依存しない独立 project です。
    - `apps/vertex-job-runner`: Vertex AI Custom Training Job を投入するための CLI（エントリポイント `vrun`）
- アーキテクチャは「project ごとの実験・学習コード」と「再利用可能な shared library」を分離する前提です。
- ディレクトリ構成の意味は次の通りです。
    - `apps/`: CLI や実行アプリケーションを置きます。現在は `vertex-job-runner` が該当します。
    - `libs/`: 複数 project で再利用する共通ライブラリを置きます。中心は `libs/ml_sandbox_libs` です。
    - `projects/`: 推薦、分類などの各 ML project を置きます。学習コード、設定、tests を project ごとに持ちます。
    - `infra/`: Terraform などのインフラ定義を置きます。Python package と同じ前提では扱いません。
    - `make/`: 全 package の Makefile が共通して include する `help.mk`, `python.mk` を置きます。
    - `docs/`: 設計資料や plan など、ドキュメント類を置きます。Python package ではありません。
- 共通化できる型、module、utility は `libs/ml_sandbox_libs` に寄せます。
- project 固有の business logic、training flow、model composition は各 project 配下に残します。
- monorepo 内の package 間依存は `pyproject.toml` の `[tool.uv.sources]` で local path を editable 指定します。recsys 系 project は `ml-sandbox-libs` と `vertex-job-runner` に依存します。`sentiment_analysis` は独立 project で shared library に依存しません。
- `libs/ml_sandbox_libs` を変更した場合は、library 単体だけでなく downstream project への影響も確認します。
- shared module の単体テストは `libs/ml_sandbox_libs/tests` に置き、project 側には integration test を残します。
- public な使い方、配置、import path を変えた場合は、関連 README も更新対象です。

## 2. Tech Stack & Libraries

- 主言語は Python です。
- Python は 3.12 系を前提にします。各 package の `pyproject.toml` で `requires-python` は 3.12 系で揃っています（`vertex-job-runner` のみ `>=3.12` で上限なし）。
- ツール管理は `mise.toml` と `uv` を前提にします。`mise.toml` で `uv` と `fd` を管理します。現状の `uv` 管理バージョンは 0.11.6 です。
- 多くの Python package は `pyproject.toml` と `Makefile` を持ち、依存管理・lint・test は package 単位で行います。各 Makefile は `make/help.mk` と `make/python.mk` を include し、共通の `lint`, `fmt`, `test`, `clean-cache`, `lock` target を提供します。
- 中心となるライブラリ群は、PyTorch、Lightning、Hydra、Polars、NumPy、TorchMetrics、Loguru、Torch Geometric (PyG)、`datasets`、`tensorboard`、`timm` です。
- project / app ごとの主な追加ライブラリ:
    - `recsys-ranking`, `recsys-candidate-generation`: `wandb`、`timm`。PyTorch は `cpu` / `gpu` の optional dependency（extra）で切り替え。`recsys-candidate-generation` は PyG 拡張（`torch_scatter`, `torch_sparse`, `torch_cluster`, `pyg_lib`）も利用。
    - `sentiment_analysis`: `transformers`、`datasets`、`spacy`、`bertviz`、`rich`、`tensorboard`（PyTorch は `lightning` 経由）。
    - `vertex-job-runner`: `google-cloud-aiplatform`、`pydantic`、`pydantic-settings`、`typer`。
    - `ml_sandbox_libs`: `google-cloud-logging`、`torch_geometric` と PyG 拡張。
- 依存追加は慎重に扱ってください。既存ライブラリで解決できるなら新規 package を増やしません。
- 新しい Python package を勝手に追加しないでください。追加が必要な場合だけ、理由が明確で影響範囲が分かる状態で対象 package の `pyproject.toml` を最小限更新します。
- 既存の依存解決や GPU/CPU 切替は `uv` と `pyproject.toml` の `optional-dependencies`（`cpu` / `gpu` extra）に従います。個別の install 手順を横に増やさないでください。
- Python の docstring は Google style を使います。`Args:`, `Returns:`, `Raises:` を使い、型注釈がある前提で docstring 内に型を重ねて書く必要はありません。

## 3. Workflow Commands

- Python 関連の実行は必ず `uv` を通します。`python`, `pip`, `pytest` を直接叩かず、`uv run ...` または各 package の `make` を使ってください。
- 作業前に、変更対象の package root へ移動してからコマンドを実行してください。
- 共通 target（`make/python.mk` 経由で全 package が利用可能）:
    - `make install`: 依存関係のセットアップ（package 固有の定義による）
    - `make fmt`: formatter の適用（`ruff check --fix` + `ruff format`）
    - `make lint`: `ruff check` と `mypy` の実行
    - `make test`: `pytest` の実行
    - `make lock`: `uv lock` の実行
    - `make clean-cache`: cache 削除
- `libs/ml_sandbox_libs` の標準確認:
    - `cd libs/ml_sandbox_libs && make install`（GPU 有無で `--extra=gpu` / `--extra=cpu` を自動切替）
    - `cd libs/ml_sandbox_libs && make fmt`
    - `cd libs/ml_sandbox_libs && make lint`
    - `cd libs/ml_sandbox_libs && make test`
- `projects/recsys-ranking` の標準確認:
    - `cd projects/recsys-ranking && make install`（GPU 有無で `--extra=gpu` / `--extra=cpu` を自動切替）
    - `cd projects/recsys-ranking && make fmt`
    - `cd projects/recsys-ranking && make lint`
    - `cd projects/recsys-ranking && make test`
    - 学習実行は `uv run python src/fit.py model=DeepFM data.batch_size=32` のように Hydra override を使います。`make train` target は未定義のため `uv run python src/fit.py ...` を直接実行してください。
- `projects/recsys-candidate-generation` も基本は同じで、package root で `make install`, `make fmt`, `make lint`, `make test` を使います。
    - 学習実行は `uv run python src/fit.py model=SASRec data.batch_size=64` のように Hydra override を使います。`make train` target は未定義のため `uv run python src/fit.py ...` を直接実行してください。
    - GPU 環境向けに `Dockerfile` と `compose.yaml` があり、`docker compose` で GPU コンテナを起動できます。
- `projects/sentiment_analysis` の標準確認:
    - `cd projects/sentiment_analysis && make install`
    - `cd projects/sentiment_analysis && make fmt`
    - `cd projects/sentiment_analysis && make lint`
    - `cd projects/sentiment_analysis && make test`
    - 学習実行は `make train`（`uv run python train.py`）。この project は `libs/ml_sandbox_libs` に依存しない独立 project です。
- `apps/vertex-job-runner` の標準確認:
    - `cd apps/vertex-job-runner && make install`
    - `cd apps/vertex-job-runner && make fmt`
    - `cd apps/vertex-job-runner && make lint`
    - `cd apps/vertex-job-runner && make test`
    - README ベースで確認する場合も `uv sync`, `uv run pytest` を使います。
- `infra/terraform` は Python package と切り離して扱い、変更時は Terraform 側の Makefile と構成を確認してから実行します。
    - ローカルの Terraform CLI 版は `infra/terraform/.terraform-version` に pin します（tfenv / tenv 向け）。CI の `hashicorp/setup-terraform` はこのファイルを自動では読まないため、workflow が中身を `terraform_version` に渡します。
    - `cd infra/terraform && make format`（`terraform fmt -recursive`）
    - `cd infra/terraform && make format-check`（`terraform fmt -check -recursive`）
    - `cd infra/terraform && make lint`（`tflint --init` + `tflint` + `trivy`）
    - `cd infra/terraform && make validate`（`terraform init -backend=false -lockfile=readonly` + `terraform validate` in `envs/dev`）
- CI: GitHub Actions（`.github/workflows/python-ci.yml`）が `pyproject.toml` + `Makefile` を持つ package を自動検出し、`make install`, `make lint`, `make test` を実行します。PR 時は各 package でこれらが通ることを前提にします。Terraform CI（`.github/workflows/terraform-ci.yml`）は path filter なしで全 PR と `main` への push で `make format-check`, `make lint`, `make validate` を実行します。
- 変更した package では少なくとも `make lint` と `make test` を通します。
- `libs/ml_sandbox_libs` を変更した場合は、必要に応じて関連 project の test も追加で実行します。

## 4. Never Do

- `python`, `pip`, `pytest` を直接実行しない。
- package root 以外で雑にコマンドを打たない。
- unrelated file を formatting や import 並び替えのためだけに広く触らない。
- shared 化が目的なのに、不要な互換レイヤや迂回 API を増やさない。
- project 内 duplicate を放置したまま、共通化すべきロジックを別名で増やさない。
- project 固有の business logic や model composition を安易に `libs` へ移さない。
- 新しい依存 package を無根拠に追加しない。
- public function / method に型注釈なしの定義を追加しない。
- `except: pass`、握り潰し、曖昧な fallback でエラーを隠さない。呼び出し側が把握すべき例外条件は `docstring` と実装の両方で明確にします。
- 関数・method を新規追加または実質変更したのに、`docstring` や関連ドキュメントを古いまま残さない。
- Python の `docstring` で最低限必要な情報を省略しない。何をするか、引数の意味と前提条件、返り値の意味、主要な例外条件を `Args:`, `Returns:`, `Raises:` で記述します。
- `libs/ml_sandbox_libs` を変更したのに downstream 影響を無視しない。
- brittle な test、内部実装に過度に依存する test、同じことを複数箇所で重複確認する test を増やしすぎない。
- `make lint` や `make test` が落ちると分かる状態のまま作業完了にしない。
