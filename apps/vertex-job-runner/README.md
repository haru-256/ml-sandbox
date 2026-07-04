# Vertex AI Custom Training Job Runner

`vertex-job-runner` は、Google Cloud Vertex AI の Custom Training Job を投入するための CLI パッケージです。  
`pyproject.toml` の `[tool.vrun]`、環境変数、CLI 引数を組み合わせて設定を管理し、実験ジョブの起動を再現しやすくします。

## できること

- Vertex AI Custom Training Job の実行
- `pyproject.toml` をベースにした設定管理
- `VRUN_` プレフィックス付き環境変数による上書き
- CLI オプションによる一時的な設定上書き
- `--dry-run` による投入前の設定確認

## パッケージ構成

```text
apps/vertex-job-runner/
├── Makefile
├── README.md
├── pyproject.toml
├── samples/
├── src/
│   └── vertex_job_runner/
│       ├── cli.py
│       ├── job.py
│       └── settings.py
├── tests/
└── uv.lock
```

## 前提

- Python 3.12
- `uv`
- Google Cloud Project
- Vertex AI Custom Training Job を実行するための認証と権限
- ジョブ実行に利用するコンテナイメージ
- 出力先となる GCS バケット

## インストール

パッケージルートで依存関係をセットアップします。

```sh
make install
```

または `uv` を直接使う場合は次の通りです。

```sh
uv sync
```

## 設定の読み込み順

設定は次の優先順位で適用されます。

1. CLI 引数
2. 環境変数
3. `pyproject.toml`

つまり、普段使う既定値は `pyproject.toml` に置き、実行ごとの差分だけを環境変数や CLI で上書きする運用ができます。

## `pyproject.toml` 設定

`pyproject.toml` に `[tool.vrun]` セクションを定義します。

```toml
[tool.vrun]
project = "your-gcp-project-id"
location = "us-central1"
image_uri = "gcr.io/your-project/your-image:latest"
gcs_uri = "gs://your-bucket/experiments/"
service_account = "service-account@your-project.iam.gserviceaccount.com"
experiment_name = "my-experiment"
command = ["uv", "run", "main.py"]

machine_type = "g2-standard-4"
accelerator_type = "NVIDIA_L4"
accelerator_count = 1
args = ["trainer.max_epochs=10", "data.num_workers=4"]
```

### 主な設定項目

- `project`: GCP project ID
- `location`: Vertex AI のリージョン
- `image_uri`: 実行するコンテナイメージ
- `gcs_uri`: Vertex AI SDK の staging bucket と base output dir に利用する GCS URI
- `service_account`: ジョブ実行に利用するサービスアカウント
- `experiment_name`: Vertex AI 上のジョブ名プレフィックス
- `command`: コンテナ内で実行するコマンド
- `machine_type`: 実行マシンタイプ
- `accelerator_type`: GPU 種別
- `accelerator_count`: GPU 数
- `args`: トレーニングジョブへ渡す追加引数

## 環境変数での上書き

環境変数は `VRUN_` プレフィックス付きで指定します。

```sh
export VRUN_PROJECT="my-project"
export VRUN_LOCATION="us-central1"
export VRUN_MACHINE_TYPE="n1-highmem-8"
export VRUN_ACCELERATOR_COUNT="2"
export VRUN_WANDB_API_KEY="your-wandb-api-key"
```

`VRUN_WANDB_API_KEY` を設定すると、Vertex AI の training container には `WANDB_API_KEY` として渡されます。`pyproject.toml` には API key を書かず、shell・CI secret・Secret Manager などから環境変数として渡してください。

なお、container 環境変数として渡された `WANDB_API_KEY` は Vertex AI の job 詳細や監査ログ上で参照可能です。より高いセキュリティが必要な場合は、Secret Manager 経由で取得するなどの代替手段を検討してください。

たとえば CI やローカル検証で project や machine type だけを差し替えたいときに便利です。

## CLI の使い方

エントリーポイントは `vrun` です。

### 基本実行

`pyproject.toml` の設定を使ってジョブを実行します。

```sh
uv run vrun run
```

### オプションを上書きして実行

一部のハードウェア設定や引数だけを変更して実行できます。

```sh
uv run vrun run \
  --machine-type n1-highmem-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --accelerator-count 2 \
  --args "trainer.max_epochs=20 data.num_workers=8"
```

### Dry Run

実際にはジョブを投入せず、読み込まれた設定内容だけを確認します。

```sh
uv run vrun run --dry-run
```

設定ファイル・環境変数・CLI 引数のマージ結果を確認したいときは、まず `--dry-run` を使うのが安全です。

## ML Sandbox での想定ユースケース

このパッケージは、`projects/recsys-ranking` や `projects/recsys-candidate-generation` などの学習ジョブを Vertex AI 上で再現性を持って実行するための共通 CLI として使うことを想定しています。

たとえば各 project 側の `pyproject.toml` に `[tool.vrun]` を定義しておくと、project ごとに既定のイメージやジョブ引数を持たせつつ、共通の CLI で投入できます。

## 開発ワークフロー

このリポジトリでは Python 関連の実行は `uv` または `Makefile` を通して行います。

パッケージルートで次を実行してください。

### 依存関係のセットアップ

```sh
make install
```

### Lint

```sh
make lint
```

### Test

```sh
make test
```

README ベースで個別に確認したい場合は次でも実行できます。

```sh
uv sync
uv run pytest
```

## テスト

CLI と設定読み込みの変更を行った場合は、少なくとも次を確認します。

```sh
make lint
make test
```

## 注意事項

- Vertex AI や GCS を利用するため、Google Cloud 側の認証設定が必要です。
- `service_account`、`project`、SDK staging/output 用の `gcs_uri` は実際の環境に合わせて設定してください。
- W&B を使う training container では、実行前に `VRUN_WANDB_API_KEY` を設定してください。これは container 内では `WANDB_API_KEY` として参照されます。
- 本番用途では `image_uri` に固定タグ付きイメージを使うと再現性を保ちやすくなります。
- 機密情報を `pyproject.toml` に直接書かないようにし、必要に応じて環境変数や Secret Manager を利用してください。

## 関連ディレクトリ

- `apps/vertex-job-runner`: Vertex AI ジョブ投入用 CLI
- `libs/ml_sandbox_libs`: 複数 project で共有する共通ライブラリ
- `projects/recsys-ranking`: ランキング学習 project
- `projects/recsys-candidate-generation`: 候補生成学習 project