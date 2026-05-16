# RecSys Candidate Generation

推薦システムにおける **Candidate Generation（候補生成 / Retrieval）** の実験コードを管理する project です。
大規模な item 群から、各 user に対して関連性の高い候補を高速に絞り込み、後段の Ranking モデルへ渡すことを目的にしています。

## Overview

大規模推薦では、すべての item をそのまま精密に順位付けすることは現実的ではありません。
そのため、一般に次のような multi-stage architecture を採用します。

1. **Candidate Generation / Retrieval**
   大量の item から、user に関連しそうな候補を高速に抽出する段階
2. **Ranking**
   抽出した候補に対して、より複雑なモデルで精密にスコアリングする段階
3. **Re-ranking**
   多様性や business rule を考慮して最終リストを調整する段階

この project は、上記のうち **Step 1: Candidate Generation** を扱います。

## Scope

この project では主に次のような candidate generation 手法を対象にします。

- **Sequential Recommendation**
    - user の時系列行動履歴から次に興味を持つ item を予測
    - 例: `SASRec`, `gSASRec`
- **Collaborative Filtering / Retrieval**
    - user-item の相互作用から user / item 表現を学習し、類似度ベースで候補を取得
    - 例: `TwoTower`, `SimpleX`
- **Graph-based Collaborative Filtering**
    - user-item bipartite graph 上で message passing を行い、高次近傍を取り込んだ retrieval を学習
    - 例: `LightGCN`, `UltraGCN`

共通化できる data preprocessing、型、optimizer、学習 utility は `libs/ml_sandbox_libs` に寄せ、
project 固有の model composition や training flow はこの directory 配下に置きます。

## Dataset

実験では主に **Amazon Reviews 2023** の recommendation dataset を利用します。

- Source: <https://amazon-reviews-2023.github.io/>
- Paper: [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952)
- Main category: `Video_Games`

`ml_sandbox_libs` 側の dataset utility を利用して、主に次の前処理を行います。

- user / item の index 化
- 低頻度 ID の `UNK` 化
- user 履歴の時系列シーケンス化
- `max_seq_len` に応じた truncate / padding
- 学習・評価用の negative sampling
- item metadata の統合

この project では project-level の DataModule factory から、model に応じて適切な共通 DataModule を選択します。

- `TwoTower` / `SASRec` / `gSASRec` / `SimpleX`
  - `ml_sandbox_libs` 側の sequential recommendation 用 DataModule を利用
- `LightGCN` / `UltraGCN`
  - `ml_sandbox_libs` 側の Amazon Reviews bipartite graph DataModule を利用

project 内では factory の切り替えのみを持ち、前処理や DataModule 本体は `ml_sandbox_libs` の共通実装を参照します。

## Implemented Models

現時点で主に次の model を扱います。

- [x] `TwoTower`
- [x] `SASRec`
- [x] `gSASRec`
- [x] `SimpleX`
- [x] `LightGCN`
- [x] `UltraGCN`

`src/models/factory.py` では設定に応じて以下の model module を生成します。

- `TwoTower`
- `SASRec`
- `gSASRec`
- `SimpleX`
- `LightGCN`
- `UltraGCN`

## Project Structure

```md
recsys-candidate-generation/
├── Dockerfile
├── Makefile
├── README.md
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── results/
│   ├── artifact/
│   ├── dataset/
│   └── logs/
└── src/
    ├── config/          # Hydra configuration
    ├── const/           # project-local constants
    ├── data/            # project-level datamodule factory
    ├── loss/            # loss factory
    ├── models/          # model implementations and factory
    ├── results/         # result helpers
    ├── tests/           # test code
    └── fit.py           # training entrypoint
```

## Dependencies

この package は次の内部 package に依存します。

- `ml-sandbox-libs`
- `vertex-job-runner`

`pyproject.toml` の `tool.uv.sources` で monorepo 内の local package を参照しています。

また、PyTorch は optional dependency として `cpu` / `gpu` extra を使い分けます。

## Setup

package root で作業してください。

```sh
make install
```

この project は `uv` を前提に依存解決と実行を行います。
直接 `python` や `pytest` を使わず、`make` または `uv run ...` を利用します。

## Training

標準の training は以下で実行します。

```sh
make train
```

Hydra override を使って model や data 設定を切り替えることもできます。

```sh
uv run python src/fit.py model=SASRec data.batch_size=64
```

例えば以下のような override が利用できます。

```sh
uv run python src/fit.py model=TwoTower
uv run python src/fit.py model=SASRec
uv run python src/fit.py model=gSASRec
uv run python src/fit.py model=SimpleX
uv run python src/fit.py model=LightGCN loss=bpr
uv run python src/fit.py model=UltraGCN
```

`UltraGCN` uses the graph datamodule for triplet sampling, but the model itself does not run message passing. It precomputes train-graph degree and item-item co-occurrence constraints from the prepared Amazon Reviews graph, then learns user/item embeddings for ANN-style retrieval.

## Configuration

training entrypoint は `src/fit.py` です。
Hydra を使って `src/config/` 配下の設定を読み込みます。

主な flow は以下です。

1. logger の初期化
2. shared datamodule の生成
3. optimizer の生成
4. model module の生成
5. Lightning trainer の生成
6. `trainer.fit(...)` の実行

GPU 実行時には matmul precision の設定も行います。

## Vertex AI

この package は `vertex-job-runner` に依存しており、Vertex AI custom training job との連携を前提にしています。
`pyproject.toml` の `[tool.vrun]` に job 実行用の設定を定義できます。

例:

```toml
[tool.vrun]
project = "haru256-ml-sandbox"
location = "us-central1"
image_uri = "gcr.io/deeplearning-platform-release/base-cpu:latest"
gcs_uri = "gs://haru256-vertex-ai-sandbox/vertex-job-runner/"
experiment_name = "vertex-job-runner-experiment"
machine_type = "g2-standard-4"
accelerator_type = "NVIDIA_L4"
accelerator_count = 1
```

実際の job 実行時は `apps/vertex-job-runner` 側の README も参照してください。

## Development Workflow

標準 workflow は次の通りです。

```sh
make install
make fmt
make lint
make test
```

変更時は少なくとも次を確認してください。

- `make lint`
- `make test`

shared library である `libs/ml_sandbox_libs` に影響する変更を伴う場合は、downstream への影響も確認します。

## Notes

- optimizer、training utility、shared data component は `ml_sandbox_libs` を利用します
- project 固有の business logic や model composition はこの project に残します
- public な使い方や import path を変えた場合は関連 README も合わせて更新します
