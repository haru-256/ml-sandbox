# ml_sandbox_libs

`ml_sandbox_libs` は、`ml-sandbox` 内の複数 project で再利用するための共通ライブラリです。  
推薦系 project を中心に、**データ前処理**, **共通モデル部品**, **optimizer**, **学習補助 utilities** を提供します。

現在は特に以下の project から利用される前提です。

- `projects/recsys-ranking`
- `projects/recsys-candidate-generation`

project 固有の business logic や training flow は各 project 側に残し、再利用可能な型・module・utility をこの package に集約します。

## 対象読者

- `ml-sandbox` 内で共通実装を追加したい人
- 推薦 project から shared module を import して使いたい人
- Amazon Reviews 2023 ベースの前処理や DataModule を再利用したい人

## ディレクトリ構成

```text
libs/ml_sandbox_libs/
├── Makefile
├── README.md
├── pyproject.toml
├── src/ml_sandbox_libs/
│   ├── data/
│   │   └── amazon_reviews_dataset/
│   ├── models/
│   │   ├── base/
│   │   └── modules/
│   ├── optimizer/
│   ├── training/
│   └── utils/
├── tests/
└── samples/
```

## 提供している主な機能

### 1. データ処理: `ml_sandbox_libs.data.amazon_reviews_dataset`

Amazon Reviews 2023 を使った推薦実験向けの共通前処理を提供します。

主な責務:

- dataset / metadata の取得
- user / item / category index の構築
- Unknown / Padding を含む feature index 管理
- sequential recommendation 用の前処理
- bipartite graph 用の前処理
- Lightning DataModule の提供

主なコンポーネント例:

- `fetch_dataset`
- `fetch_metadata`
- `common_preprocess_dataset`
- `seq_rec_preprocess_dataset`
- `AmazonReviewsSeqRecDataset`
- `AmazonReviewsSeqRecDataModule`
- `AmazonReviewsBipartiteGraphDataModule`

想定ユースケース:

- `recsys-ranking` でのランキング学習用データ供給
- `recsys-candidate-generation` での sequential / retrieval 系学習用データ供給
- graph ベース推薦実験のための heterogeneous graph 構築

### 2. 共通モデル基盤: `ml_sandbox_libs.models`

複数 project で使い回すためのモデル基盤と共通 module を提供します。

#### `models.base`

- `BaseModule`: 推薦モデル用 LightningModule の共通基底 interface

#### `models.modules`

再利用可能な NN 部品をまとめています。例:

- `IdEmbedding`
- `LinearBlock`
- `MLP`
- `DINAttention`
- `FeatureEmbeddingDict`
- `Dice`
- その他 base / recommendation module 群

これらは ranking 系モデルや candidate generation 系モデルでの部品再利用を目的としています。

### 3. Optimizer 関連: `ml_sandbox_libs.optimizer`

optimizer と learning rate scheduler を共通化しています。

主なコンポーネント:

- `Optimizer` protocol / interface
- `AdamWCosine`
- `create_optimizer`
- optimizer 関連型定義

学習 project 側では、optimizer 実装を個別に持たず、この package の共通実装を利用する前提です。

### 4. 学習補助: `ml_sandbox_libs.training`

training 時の共通ロジックを提供します。

主なコンポーネント例:

- `ExperimentMonitor`

役割:

- logging の統一
- 学習 loop 周辺の共通的な責務の集約
- project 間での metric / monitor 実装の重複削減

### 5. Loss: `ml_sandbox_libs.loss`

loss 関連の protocol と実装を提供します。

主なコンポーネント例:

- `ScoreLossFn`
- `EmbeddingLossFn`
- `BCE`
- `gBCE`
- `CCL`

### 6. Utilities: `ml_sandbox_libs.utils`

汎用 utility 群です。

主な内容:

- logger 設定
- 類似度計算
- metrics 補助
- その他共通 helper

## 利用例

### Amazon Reviews の sequential recommendation 用 DataModule を使う

```python
from pathlib import Path

from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecDataModule

datamodule = AmazonReviewsSeqRecDataModule(
    save_dir=Path("results/dataset"),
    batch_size=32,
    num_workers=2,
    max_seq_len=50,
    neg_sample_size=1,
)
```

### 共通 optimizer を使う

```python
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.optimizer.types import LRSchedulerParams

optimizer = AdamWCosine(
    lr=1e-3,
    weight_decay=1e-2,
    lr_scheduler_params=LRSchedulerParams(
        step_unit="epoch",
        t_initial=100,
        warmup_t=5,
        warmup_lr_init=1e-5,
        lr_min=1e-6,
        frequency=1,
        cycle_limit=1,
    ),
)
```

### 共通 monitor を使う

```python
import lightning as L

from ml_sandbox_libs.training import ExperimentMonitor


class MyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.monitor = ExperimentMonitor(self)

    def training_step(self, batch, batch_idx):
        loss = ...
        self.monitor.logging_step({"loss": loss}, stage="train", batch_idx=batch_idx)
        return loss
```

## セットアップ

この package は Python 3.12 系を前提としています。

package root に移動してセットアップします。

```sh
cd libs/ml_sandbox_libs
make install
```

`uv` 管理を前提としており、依存関係は `pyproject.toml` と `uv.lock` に従います。

## 開発フロー

作業は必ず package root で実行します。

```sh
cd libs/ml_sandbox_libs
make fmt
make lint
make test
```

標準コマンド:

- `make install`: 依存関係のセットアップ
- `make fmt`: formatter の適用
- `make lint`: `ruff` / `mypy`
- `make test`: `pytest`

## shared library としての運用方針

この library を変更するときは、以下を意識してください。

- 共通化できる型・module・utility のみを置く
- project 固有の business logic は各 project 配下に残す
- public な import path を変えた場合は downstream project も更新する
- `recsys-ranking` / `recsys-candidate-generation` への影響を確認する
- shared module の unit test は `libs/ml_sandbox_libs/tests` に置く
- 関数や class を追加・変更した場合は docstring も更新する

## 関連 project

- `../../projects/recsys-ranking`
- `../../projects/recsys-candidate-generation`
- `../../apps/vertex-job-runner`

## Notes

- Amazon Reviews 2023 関連の共通処理は、この library 側に寄せています。
- optimizer / model module / training utility の shared 化を進めることで、project 間の重複実装を減らしています。
- downstream project で import path を変更した場合は README やサンプルコードも合わせて更新してください。
