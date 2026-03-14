# ml_sandbox_libs

複数の推薦システムプロジェクト間で共有するユーティリティ・共通コンポーネントのライブラリです。

## モジュール構成

```
src/ml_sandbox_libs/
├── my_types.py          # 共通型定義（LRSchedulerParams, OptimizerParams）
├── optimizer/           # Optimizer 実装
│   ├── base.py          # Optimizer プロトコル
│   ├── adam_w_cosine.py # AdamW + CosineLR スケジューラ
│   └── factory.py       # create_optimizer ファクトリ関数
├── training/
│   └── monitor.py       # ExperimentMonitor (ログ・モニタリング)
├── data/                # データセット・DataModule
├── utils/               # 汎用ユーティリティ
│   ├── metrics.py       # 検索・分類メトリクス
│   ├── similarity.py    # 類似度計算
│   └── utils.py         # ロガー設定など
└── tests/               # テスト
```

## インストール

```sh
# CPU 環境
make install

# GPU 環境（CUDA 検出時は自動的に GPU 向け PyTorch を使用）
make install
```

または手動で:

```sh
uv venv && source .venv/bin/activate
uv pip install -e .
```

## 主要コンポーネント

### 型定義 (`optimizer.types`)

```python
from ml_sandbox_libs.optimizer.types import LRSchedulerParams, OptimizerParams
```

### Optimizer

```python
from ml_sandbox_libs.optimizer import AdamWCosine, Optimizer, create_optimizer

optimizer = AdamWCosine(
    lr=1e-3,
    weight_decay=1e-2,
    lr_scheduler_params=LRSchedulerParams(
        step_unit="epoch", t_initial=100, warmup_t=5,
        warmup_lr_init=1e-5, lr_min=1e-6, frequency=1, cycle_limit=1,
    ),
)
```

### ExperimentMonitor

Lightning モジュール内でのロギングを統一する `ExperimentMonitor`:

```python
from ml_sandbox_libs.training import ExperimentMonitor

class MyModule(L.LightningModule):
    def __init__(self, ...):
        self.monitor = ExperimentMonitor(self)

    def training_step(self, batch, batch_idx):
        ...
        self.monitor.logging_step({"loss": loss}, stage="train", batch_idx=batch_idx)
```

## 開発

```sh
make lint   # ruff + mypy
make fmt    # ruff format
make test   # pytest
```
