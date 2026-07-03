# RecSys Ranking

推薦システムにおける **Ranking** 段階の実験コードを管理する project です。
Candidate Generation で絞り込まれた候補アイテムに対して、ユーザーの興味や文脈に基づく精緻なスコアリングを行い、最終的な表示順位を決定します。

## Overview

多段推薦アーキテクチャでは、一般に次の流れで推薦を行います。

1. **Candidate Generation**: 大量のアイテム集合から、ユーザーに関連しそうな候補を高速に抽出する
2. **Ranking**: 抽出された候補をより表現力の高いモデルで並び替える
3. **Re-ranking**: 多様性やビジネスルールなどを考慮して最終調整する

この project は上記のうち **Step 2: Ranking** に対応しています。

## Project Scope

`projects/recsys-ranking` では、主に以下を扱います。

- Amazon Reviews 2023 を用いたランキング実験
- Hydra ベースの学習設定管理
- PyTorch / Lightning による学習ループ
- 共通 DataModule / optimizer / utility を `ml_sandbox_libs` から利用した構成
- Deep learning ベースの推薦モデル比較

shared 化できる型や utility は `libs/ml_sandbox_libs` に寄せ、ranking 固有の model composition や training flow はこの project 配下で管理します。

## Dataset

実験では主に **Amazon Reviews 2023** を使用します。

- Source: <https://amazon-reviews-2023.github.io/>
- Paper: [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952)

ランキング学習では、Candidate Generation で得られた候補をより精密に判別するために、ユーザー履歴とターゲットアイテムの関係をモデリングします。
本 project では、`ml_sandbox_libs` が提供する Amazon Reviews 向け前処理・DataModule を活用しています。
DataModule の生成処理は project 内に重複実装せず、shared library 側の factory / DataModule を利用する前提です。

## Directory Structure

```text
recsys-ranking/
├── Makefile              # package 単位の開発コマンド
├── pyproject.toml        # 依存関係・tool 設定
├── README.md             # このファイル
├── uv.lock               # lock file
├── examples/             # 実行例・補助資料
├── results/              # 実験結果、ログ、アーティファクト保存先
└── src/
    ├── fit.py            # 学習エントリポイント
    ├── config/           # Hydra 設定
    ├── const/            # project 固有定数
    ├── data/             # namespace package（DataModule 実体は shared library を利用）
    ├── loss/             # loss factory
    ├── models/           # ranking model 群
    ├── results/          # project 固有の結果処理
    ├── utils/            # project 固有 utility
    └── tests/            # test code
```

## Implemented Models

現時点で README と実装から確認できる主なモデルは以下です。

- `DeepFM`
- `DLRM`
- `DIN`
- `DCNv2`

README 上では今後の候補として以下も言及できます。

- `DCN`
- `FinalNet`

ただし、日常的に参照すべき正確な実装状況は `src/models` を基準に確認してください。

### DeepFM

Factorization Machine による低次の特徴量相互作用と、MLP による高次の非線形相互作用を組み合わせるモデルです。

- 履歴アイテムとターゲットアイテムを shared embedding で表現
- FM branch で低次相互作用を学習
- Deep branch で高次の相互作用を学習
- 両者を統合して ranking logit を出力

Reference: <https://arxiv.org/abs/1703.04247>

### DLRM

Dense / sparse feature を分けて扱い、特徴量相互作用層で結合する recommendation model です。

- sparse categorical feature の embedding
- dense numerical feature の MLP 変換
- interaction layer による feature combination
- top MLP による最終予測

Reference: <https://arxiv.org/abs/1906.00091>

### DIN

**Deep Interest Network** は、ターゲットアイテムに条件づけた attention によって、ユーザー履歴から関心表現を動的に抽出します。

- 履歴とターゲットの embedding を個別に表現
- target-aware attention で履歴の重要度を推定
- 重み付き集約表現を使ってクリック確率を予測

Reference: <https://arxiv.org/abs/1706.06978>

### DCNv2

**Deep & Cross Network V2** の parallel variant を採用し、cross network と deep network を併用して明示的・暗黙的な特徴量相互作用を同時に学習します。

- cross network による明示的 feature interaction
- MLP による高次の非線形表現
- behavior encoder と組み合わせた履歴集約
- ranking 向けの最終 logit 出力

Reference: <https://arxiv.org/abs/2008.13535>

## Training Workflow

この repository では Python 実行を `uv` ベースで統一しています。
作業時は必ず package root である `projects/recsys-ranking` に移動してからコマンドを実行します。

### Setup

```sh
make install
```

### Format

```sh
make fmt
```

### Lint

```sh
make lint
```

### Test

```sh
make test
```

変更時は少なくとも `make lint` と `make test` を通す想定です。

## Training

基本の学習実行は次の通りです。

```sh
uv run python src/fit.py
```

Hydra の override を使って個別設定を変更することもできます。

```sh
uv run python src/fit.py model=DeepFM data.batch_size=32
```

モデル切り替え例:

- `model=DeepFM`
- `model=DLRM`
- `model=DIN`
- `model=DCNv2`

使用可能な設定名は `src/config` と `src/models/factory` の対応に従います。

### Checkpoint

model checkpoint は default では保存しません。
必要な場合は `enable_checkpointing=true` を指定すると、`save_dir` 配下の `checkpoints/` に保存します。

```sh
uv run python src/fit.py enable_checkpointing=true
```

## Local Docker

Dockerfile から image を build して、ローカルで学習を実行できます。

### 前提

- Docker（GPU アクセスのため `--gpus all` をサポート）
- `WANDB_API_KEY` 環境変数（[wandb.ai/authorize](https://wandb.ai/authorize) から取得）
- Dev shell では `WANDB_API_KEY` を `.env` に設定するか、`make docker-up` 実行時の shell で export してください。

### image を build して学習を直接実行

```sh
export WANDB_API_KEY=your-api-key
make docker-build
make docker-run
```

Hydra override も渡せます:

```sh
make docker-run DOCKER_ARGS="model=DeepFM data.batch_size=32"
```

### Dev shell としての利用

```sh
cp .env.example .env
# .env の WANDB_API_KEY を設定
make docker-up
make docker-exec
make docker-down
```

### 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `WANDB_API_KEY` | yes | wandb 認証用 API key |
| `IMAGE_URI` | optional | Docker image 名（default: `recsys-ranking:latest`。`make docker-run` では `DOCKER_IMAGE` 変数を使用） |
| `EXPERIMENT_NAME` | optional | Cloud Logging 用（ローカル Docker では使用されない） |

## Dependencies

この package は主に以下の依存を利用します。

- `lightning`
- `torchmetrics`
- `numpy`
- `polars`
- `hydra-core`
- `wandb`
- `timm`
- `ml-sandbox-libs`

また、`cpu` / `gpu` の optional dependency を通じて PyTorch と torchvision を切り替える構成です。
Vertex AI custom training job 連携が必要な場合は `vertex` dependency group で `vertex-job-runner` を追加します。

## Relationship with Shared Libraries

共通化されたコンポーネントは主に `libs/ml_sandbox_libs` と `apps/vertex-job-runner`（`vertex` dependency group）から参照します。

- `ml_sandbox_libs`: DataModule、shared datamodule factory、共通 model module、optimizer、training utility
- `vertex-job-runner`: Vertex AI 上での job 実行補助

特に、型定義・optimizer・monitoring などの shared module は project 内に重複実装せず、共通 library から import する前提です。

## Notes

- 実装の正確な現状は README よりも `src/` 配下のコードを優先してください。
- ranking 固有の business logic はこの project に残し、他 project でも再利用する utility のみ `libs/ml_sandbox_libs` に寄せます。
- public な使い方や import path を変更した場合は、関連 README も合わせて更新してください。
