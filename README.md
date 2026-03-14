# ML-Sandbox

[![Python CI](https://github.com/haru-256/ml-sandbox/actions/workflows/python-ci.yml/badge.svg)](https://github.com/haru-256/ml-sandbox/actions/workflows/python-ci.yml)

機械学習モデルの実装・実験・再利用可能な基盤整備を進めている Python モノレポです。  
特に推薦システムを中心に、Candidate Generation、Ranking、共通ライブラリ化、実験実行基盤までを扱っています。

この README は、リポジトリで扱っている問題設定や実装方針を俯瞰できる概要資料として書いています。  
どのような問題設定を扱っているか、どのような実装をしているか、どのような設計で整理しているかが伝わることを目的にしています。

## このリポジトリで伝えたいこと

このリポジトリでは、主に次のようなテーマを扱っています。

- 推薦システム
  - Multi-Stage Recommendation Architecture
  - Candidate Generation / Retrieval
  - Ranking
  - Sequential Recommendation
  - CTR / CVR を意識した特徴量相互作用モデリング
- 機械学習実験基盤
  - Hydra を使った設定管理
  - PyTorch / Lightning を使った学習ループ
  - 共通ライブラリ化による再利用性向上
  - Vertex AI へのジョブ投入
- データ処理
  - Amazon Reviews 2023 を使った推薦用前処理
  - user / item / category index 構築
  - sequence 化
  - negative sampling
  - metadata の統合

このリポジトリは、単にモデルを個別実装するだけでなく、問題設定ごとの project 分離、共通ライブラリ化、クラウド実行基盤の整備まで含めて設計しています。

## リポジトリ構成

```text
.
├── README.md
├── apps/                      # 実行アプリケーション・CLI
│   └── vertex-job-runner
├── libs/                      # 複数 project で再利用する内部ライブラリ
│   └── ml_sandbox_libs
├── projects/                  # 問題設定ごとの ML project
│   ├── recsys-candidate-generation
│   ├── recsys-ranking
│   └── sentiment_analysis
└── infra/                     # Terraform などのインフラ定義
```

## 推薦システム全体アーキテクチャ

推薦システム全体の問題設定として、以下のような multi-stage architecture を前提にしています。

```mermaid
flowchart LR
    U[User / Context] --> CG[Candidate Generation]
    I[Item Corpus] --> CG
    CG --> C[Candidate Set]
    C --> R[Ranking]
    U --> R
    M[Item Metadata / Features] --> R
    R --> RR[Re-ranking / Business Rules]
    RR --> O[Final Recommendation List]
    O --> S[Serving / UI]

    classDef stage fill:#e8f0fe,stroke:#4c6ef5,color:#111,stroke-width:1.5px;
    classDef data fill:#eefbee,stroke:#2f9e44,color:#111,stroke-width:1.5px;
    class CG,R,RR stage;
    class U,I,C,M,O,S data;
```

### なぜこれが重要か

大規模推薦では、全 item を精密に score することは計算量的に難しいため、通常は以下の責務分離が必要になります。

- Candidate Generation
  - 大規模 item 集合から、関連性の高そうな候補を高速に絞る
  - 重要なのは recall を落としすぎないこと
- Ranking
  - 候補集合に対して、より表現力の高いモデルで精密にスコアリングする
  - 特徴量相互作用や target-aware modeling が重要
- Re-ranking
  - 多様性、在庫、ビジネスルール、露出制御などを反映する

このリポジトリでは、特に Candidate Generation と Ranking を別 project として切り出し、問題設定と設計判断を分離しています。

## 自作の推薦システムアーキテクチャ図

推薦システム全体像を、自分の実装関心に寄せて表現すると次のようになります。  
この図では、大規模 item corpus からの候補生成、特徴量を使った ranking、business rule を反映する re-ranking、serving までを 1 本の流れとして整理しています。

```mermaid
flowchart LR
    subgraph Online["Online / Serving Path"]
        U[User Request]
        UC[User Context<br/>profile / session / history]
        CG[Candidate Generator<br/>TwoTower / SASRec / gSASRec / SimpleX]
        CS[Candidate Set]
        RK[Ranker<br/>DeepFM / DLRM / DIN / DCNv2]
        RR[Re-ranker<br/>diversity / rules / filtering]
        OUT[Top-N Recommendations]
    end

    subgraph FeatureStore["Feature / Metadata Layer"]
        IM[Item Metadata<br/>category / attributes / statistics]
        UF[User Features]
        CF[Context Features]
    end

    subgraph Offline["Offline / Training Path"]
        RAW[Amazon Reviews 2023]
        PRE[Shared Preprocessing<br/>index build / sequence build / negative sampling]
        DM[DataModule / Dataset]
        TRCG[Candidate Generation Training]
        TRRK[Ranking Training]
        REG[Model Registry / Artifacts]
    end

    RAW --> PRE
    PRE --> DM
    DM --> TRCG
    DM --> TRRK
    TRCG --> REG
    TRRK --> REG

    U --> UC
    UC --> CG
    IM --> CG
    CG --> CS

    CS --> RK
    UC --> RK
    UF --> RK
    CF --> RK
    IM --> RK

    RK --> RR
    RR --> OUT

    REG -. deploy .-> CG
    REG -. deploy .-> RK
```

### 図の見方

- Offline / Training Path
  - Amazon Reviews 2023 を共通前処理し、Candidate Generation 用・Ranking 用の学習データを構築
  - 学習済みモデルや artifact を管理し、online path に deploy
- Candidate Generator
  - 大規模 corpus から候補を高速に絞る層
  - recall を重視し、user/item representation learning や sequence modeling が中心
- Ranker
  - 候補集合に対して、特徴量相互作用を用いて score する層
  - user history、target item、metadata、dense/sparse features を統合
- Re-ranker
  - ビジネスルール、多様性、フィルタリングなどを反映して最終リストを作る層
- Serving
  - online request に対して段階的に candidate を絞り、最終的な Top-N を返す層

### この図を入れている理由

この図は、このリポジトリが単なるモデル実装集ではなく、

- retrieval と ranking の責務分離
- offline training と online serving の接続
- 共通前処理と feature 再利用
- 実験コードと実運用寄り設計の橋渡し

まで意識していることを示すために入れています。

## モノレポ全体の構成

```mermaid
flowchart LR
    DS[Amazon Reviews 2023]

    subgraph LIBS[共通ライブラリ]
        PRE[前処理・DataModule<br/>libs/ml_sandbox_libs]
        MOD[共通 model module / optimizer / utility<br/>libs/ml_sandbox_libs]
    end

    subgraph PROJECTS[各 project]
        CG[Candidate Generation<br/>projects/recsys-candidate-generation]
        RK[Ranking<br/>projects/recsys-ranking]
    end

    subgraph APPS[実行レイヤ]
        VR[Vertex AI Job Runner<br/>apps/vertex-job-runner]
    end

    OUT[Candidate Items]
    REC[Ranked Recommendations]

    DS --> PRE
    PRE --> CG
    PRE --> RK
    MOD --> CG
    MOD --> RK

    CG --> OUT
    OUT --> RK
    RK --> REC

    VR --> CG
    VR --> RK
```

この図では、データセット、共通ライブラリ、各 project、実行レイヤの関係を分けて示しています。

- `libs/ml_sandbox_libs` は前処理・DataModule・共通 module を提供
- `projects/recsys-candidate-generation` と `projects/recsys-ranking` はそれらを利用して学習・実験を実施
- `apps/vertex-job-runner` は各 project の実行を補助
- Candidate Generation の出力が Ranking の入力になる

## projects

ここがこのリポジトリの中心です。  
特に推薦システム領域において、検索・候補生成・ランキングという構成に近い問題設定を扱っています。

---

## Recsys Candidate Generation

`projects/recsys-candidate-generation`

推薦システムにおける Candidate Generation（候補生成 / Retrieval）を扱う project です。  
大規模な item 集合から、各 user に対して関連性の高い候補を高速に絞り込む段階を対象にしています。

### 技術的な焦点

- Sequential Recommendation
  - user の時系列行動履歴から次に関心を持つ item を予測
- Collaborative Filtering / Retrieval
  - user-item 相互作用から user / item representation を学習し、候補を取得
- Negative Sampling
  - 正例と負例の対比による efficient training
- Representation Learning
  - retrieval quality を高めるための embedding 学習

### 実装しているモデル

- `TwoTower`
- `SASRec`
- `gSASRec`
- `SimpleX`

### Candidate Generation モデル比較

| Model | Category | Main Input | Core Idea | Strength | Typical Use Case |
|---|---|---|---|---|---|
| `TwoTower` | Retrieval / dual-encoder | user ID, item ID, optional user/item features | user tower と item tower を独立に学習し、embedding 類似度で候補取得 | serving しやすい、ANN と相性が良い、実運用に載せやすい | large-scale retrieval |
| `SASRec` | Sequential Recommendation | user interaction sequence | self-attention で行動履歴の依存関係を表現し、次 item を予測 | sequence の文脈を捉えやすい | next-item prediction |
| `gSASRec` | Sequential Recommendation | user interaction sequence | SASRec に gBCE 系の考え方を導入し、negative sampling 起因の過信を抑制 | hard negative に対する学習安定性を意識できる | robust sequential retrieval |
| `SimpleX` | Collaborative Filtering | user history, item interactions | user history の単純集約と contrastive 的な学習を組み合わせる | 実装が比較的シンプルで強い baseline になりやすい | strong CF baseline |

### この project の技術的な意味

この project では、単に推薦モデルを実装しただけではなく、以下の論点を扱っています。

- retrieval と ranking は目的関数も制約も異なる
- sequence modeling は retrieval quality に直接効く
- embedding 学習は ANN 検索や serving 設計と接続しやすい
- negative sampling の設計は性能と学習安定性に強く影響する

### 参考論文

| Topic / Model | Paper | Link |
|---|---|---|
| Dataset | Bridging Language and Items for Retrieval and Recommendation | <https://arxiv.org/abs/2403.03952> |
| SASRec | Self-Attentive Sequential Recommendation | <https://arxiv.org/abs/1808.09781> |
| gSASRec | Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling | <https://arxiv.org/abs/2308.07192> |
| SimpleX | SimpleX: A Simple and Strong Baseline for Collaborative Filtering | <https://arxiv.org/abs/2109.12613> |

詳細は [`projects/recsys-candidate-generation/README.md`](projects/recsys-candidate-generation/README.md) を参照してください。

## Recsys Ranking

`projects/recsys-ranking`

推薦システムにおける Ranking 段階を扱う project です。  
Candidate Generation で取得した候補 item に対して、より表現力の高いモデルで精密に順位付けする段階を対象にしています。

### Ranking 段階のイメージ

```mermaid
flowchart LR
    A[User History / Context] --> B[Candidate Items]
    B --> C[Feature Interaction Modeling]
    A --> C
    M[Metadata / Dense-Sparse Features] --> C
    C --> D[Scoring Function]
    D --> E[Sorted Candidate List]
```

Ranking では、candidate generation よりも豊かな特徴量相互作用を扱えるため、

- user の履歴
- target item
- category などの side information
- dense / sparse feature

を統合して、クリックや購買に近い signal を学習しやすくなります。

### 技術的な焦点

- CTR / ranking 向け deep models
- Feature Interaction Modeling
- Behavior Sequence Aggregation
- Attention-based User Interest Modeling
- Cross Network と MLP の比較
- Hydra を使った実験設定の切り替え

### 実装しているモデル

- `DeepFM`
- `DLRM`
- `DIN`
- `DCNv2`

### Ranking モデル比較

| Model | Category | Main Input | Core Idea | Strength | Typical Use Case |
|---|---|---|---|---|---|
| `DeepFM` | CTR / ranking | sparse categorical features, item history | FM で低次相互作用、DNN で高次相互作用を同時に扱う | wide & deep 系の代表、実装と解釈のバランスが良い | baseline ranking |
| `DLRM` | Recommendation ranking | dense + sparse features | sparse embedding と dense MLP を分離し、interaction layer で統合 | 実務で使いやすい構成、feature engineering と相性が良い | production-style ranking |
| `DIN` | Interest modeling | user behavior sequence, target item | target-aware attention により、ターゲットごとに user interest を動的抽出 | 行動履歴の relevance を item ごとに変えられる | personalized CTR prediction |
| `DCNv2` | Feature interaction | sparse / dense features, history representation | cross network と deep network を併用して explicit / implicit interaction を同時に学習 | cross feature を強く扱える、強力な ranking model | advanced feature interaction modeling |

### この project の技術的な意味

この project では、単純な分類モデルではなく、推薦特有の

- 履歴系列の扱い
- target-aware attention
- feature crossing
- sparse / dense feature の統合

といった、広告・EC・推薦において実務的に重要な論点を実装対象にしています。

### 参考論文

| Topic / Model | Paper | Link |
|---|---|---|
| DeepFM | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | <https://arxiv.org/abs/1703.04247> |
| DLRM | Deep Learning Recommendation Model for Personalization and Recommendation Systems | <https://arxiv.org/abs/1906.00091> |
| DIN | Deep Interest Network for Click-Through Rate Prediction | <https://arxiv.org/abs/1706.06978> |
| DCNv2 | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems | <https://arxiv.org/abs/2008.13535> |

詳細は [`projects/recsys-ranking/README.md`](projects/recsys-ranking/README.md) を参照してください。

---

## 共通ライブラリ

### ml_sandbox_libs

`libs/ml_sandbox_libs`

複数 project で再利用するための内部 library です。  
このリポジトリの設計思想をよく表しているディレクトリのひとつです。

### 役割

- Amazon Reviews 2023 の共通前処理
- user / item / category index の構築
- sequential recommendation 用 DataModule
- graph recommendation に拡張可能な preprocessing
- recommendation 向け共通 model module
- optimizer / lr scheduler
- training utilities

### なぜ重要か

この library があることで、

- project ごとに同じ preprocessing を書き直さない
- optimizer や training monitor の実装が重複しない
- recommendation 実験を増やしても codebase が崩れにくい

というメリットがあります。

継続的に ML project を増やしていく前提の構造になっています。

詳細は [`libs/ml_sandbox_libs/README.md`](libs/ml_sandbox_libs/README.md) を参照してください。

## 実験実行レイヤ

### vertex-job-runner

`apps/vertex-job-runner`

Google Cloud Vertex AI の Custom Training Job を実行するための CLI です。  
実験コードそのものではなく、実験をクラウド上で再現性高く実行するための基盤です。

### できること

- `pyproject.toml` の `[tool.vrun]` を使った設定管理
- 環境変数と CLI option による上書き
- dry-run による job 設定の確認
- project ごとの training entrypoint の共通実行

### なぜ重要か

このリポジトリは、モデル実装だけで終わらず、

- 実験設定
- 再現可能な job submission
- local と cloud execution の橋渡し

までを意識しています。

詳細は [`apps/vertex-job-runner/README.md`](apps/vertex-job-runner/README.md) を参照してください。

---

## 実験フロー

```mermaid
flowchart TD
    A[Dataset Fetch] --> B[Preprocessing]
    B --> C[DataModule]
    C --> D[Model Construction]
    D --> E[Optimizer / Scheduler]
    E --> F[Trainer]
    F --> G[Local Experiment]
    F --> H[Vertex AI Job]
```

このフローは、`projects/*` にある training code と、`libs/*`, `apps/*` の役割分担を要約したものです。

## 技術スタック

### 言語

- Python 3.12

### ML / DL

- PyTorch
- Lightning
- TorchMetrics
- timm

### 設定管理 / 実験管理

- Hydra
- `pyproject.toml`

### データ処理

- Polars
- NumPy
- datasets

### 開発ツール

- uv
- Makefile
- Ruff
- mypy
- pytest

### クラウド / インフラ

- Google Cloud Vertex AI
- Terraform

---

## 設計方針

このリポジトリでは、次のような方針を取っています。

### 1. project ごとの関心を分離する

- Candidate Generation と Ranking は分ける
- project 固有の model composition は project 側に置く
- shared 化できるものだけ `libs` に置く

### 2. 実験コードを再利用可能な資産として扱う

- DataModule を shared 化する
- optimizer / utility を共通化する
- import path と package 構造を整理する

### 3. ローカル実験とクラウド実行を両立する

- local package を monorepo 内で参照
- `uv` と `Makefile` で package 単位に操作
- Vertex AI 実行を CLI に分離

## 関連 README

- [`projects/recsys-candidate-generation/README.md`](projects/recsys-candidate-generation/README.md)
- [`projects/recsys-ranking/README.md`](projects/recsys-ranking/README.md)
- [`projects/sentiment_analysis/README.md`](projects/sentiment_analysis/README.md)
- [`libs/ml_sandbox_libs/README.md`](libs/ml_sandbox_libs/README.md)
- [`apps/vertex-job-runner/README.md`](apps/vertex-job-runner/README.md)

## まとめ

このリポジトリでは、推薦システムを中心に、

- Candidate Generation
- Ranking
- 共通ライブラリ化
- 実験実行基盤

といった要素を、project ごとに分けて整理しています。

個別のモデル実装だけでなく、前処理、学習コードの再利用、実験実行まで含めて扱っている点が、このリポジトリの特徴です。