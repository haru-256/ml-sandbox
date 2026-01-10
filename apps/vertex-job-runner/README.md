# Vertex AI Custom Training Job Runner

Google Cloud Vertex AI 上で機械学習のトレーニングジョブを実行するためのコマンドラインユーティリティです。
`pyproject.toml`、環境変数、CLI引数を通じて設定を管理し、カスタムコンテナトレーニングジョブの投入プロセスを簡素化します。

## 特徴

- **設定管理**: `pyproject.toml`、環境変数（`VRUN_` プレフィックス）、またはCLIフラグでジョブ設定を一元管理できます。
- **Dry Run モード**: ジョブを投入する前に、設定内容を確認することができます。
- **リッチな出力**: 設定確認用の見やすいフォーマットで出力されます。

## 設定

設定は以下の優先順位（高い順）で読み込まれます：
1. CLI 引数
2. 環境変数
3. `pyproject.toml`

### 1. pyproject.toml

`pyproject.toml` に `[tool.vrun]` セクションを追加してください：

```toml
[tool.vrun]
# 必須設定
project = "your-gcp-project-id"
location = "us-central1"
image_uri = "gcr.io/your-project/your-image:latest"
gcs_uri = "gs://your-bucket/experiments/"
service_account = "service-account@your-project.iam.gserviceaccount.com"
experiment_name = "my-experiment"
command = "python -m my_module.train"

# デフォルトのハードウェア設定（CLIで上書き可能）
machine_type = "n1-standard-4"
accelerator_type = "NVIDIA_TESLA_T4"
accelerator_count = 1

# トレーニングスクリプトへのデフォルト引数
args = ["epochs=10", "batch_size=32"]
```

### 2. 環境変数

変数の前に `VRUN_` を付けてください。例：

```sh
export VRUN_PROJECT="my-project"
export VRUN_MACHINE_TYPE="n1-highmem-8"
```

## 使い方

`uv` を使用してツールを実行するか、インストールして使用できます。

### 基本的な使い方

設定ファイルのデフォルト値を使用してジョブを投入します：

```sh
uv run vrun run
```

### 設定の上書き

特定の項目をCLIフラグで上書きします：

```sh
uv run vrun run \
  --machine-type n1-highmem-8 \
  --accelerator-count 2 \
  --args "epochs=50 learning_rate=0.001"
```

### Dry Run (確認モード)

ジョブを投入せずに設定内容を確認します：

```sh
uv run vrun run --dry-run
```

## 開発

### セットアップ

```sh
uv sync
```

### テストの実行

```sh
uv run pytest
```
