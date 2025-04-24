# RecSys Candidate Generation

## Overview

このリポジトリには、Candidate Generationの実験コードが含まれています。

Candidate Generationとは、以下2段階の推薦のMulti-Stage Architectureの1つ目の段階を指します。

1. Candidate Generation: 推薦候補の生成
2. Ranking: 候補のランキング

## データセット

データセットは、[Amazon Review 2023](https://recsys-challenge.org/) のデータセットを使用します。
Amazon Reviews dataset is large-scale dataset collected in 2023 by McAuley Lab, and it includes rich features such as:

- User Reviews (ratings, text, helpfulness votes, etc.);
- Item Metadata (descriptions, price, raw image, etc.);
- Links (user-item / bought together graphs).

ユーザーに対して、前期間にレビューしたアイテムから、次の期間中にレビューを行うアイテムを推薦することを目的としています。

related information

- HP: <https://amazon-reviews-2023.github.io/>
- paper: [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952)

## ディレクトリ構成

```sh
recsys-candidate-generation/
├── compose.vertexai.yaml   # Vertex AI 用の Docker Compose 設定
├── Dockerfile.vertexai     # Vertex AI 用の Dockerfile
├── Makefile                # ビルドやテストなどのコマンド定義
├── pyproject.toml          # Python プロジェクト設定 (依存関係など)
├── README.md               # このファイル
├── uv.lock                 # uv ロックファイル
├── results/                # 実験結果 (ログ、アーティファクトなど)
│   ├── artifact/           # モデルのアーティファクトなど
│   ├── dataset/            # データセット関連 (キャッシュなど)
│   └── logs/               # トレーニングログ
└── src/                    # ソースコード
    ├── fit.py              # トレーニングスクリプト
    ├── config/             # 設定ファイル (Hydra など)
    ├── const/              # 定数定義
    ├── models/             # モデル定義
    ├── my_types/           # 型定義
    └── tests/              # テストコード
```

## モデル

具体的には以下のモデルを実装する予定です。

- [x] TwoTower: Two-Tower Model
  - ユーザーとアイテムを独立したタワー（ニューラルネットワーク）でエンベディングし、その類似度（例：内積）を計算して推薦を行うモデル。
- [ ] MF: Matrix Factorization
- [ ] Collaborative Filtering
- [ ] NCF: Neural Collaborative Filtering
- [ ] NeuMF: Neural Matrix Factorization
- [ ] NGCF: Neural Graph Collaborative Filtering
- [ ] LightGCN
- [ ] GRU4Rec: Gated Recurrent Unit for Sequential Recommendation
- [x] SASRec: Self-Attentive Sequential Recommendation
  - TransformerのSelf-Attention機構を利用して、ユーザーの行動履歴のシーケンシャルなパターンを捉え、次のアイテムを予測するモデル。
- [ ] BERT4Rec: BERT for Sequential Recommendation
- [ ] gSASRec

### TwoTower

- **ファイルパス:** `src/models/two_tower.py`
- ユーザーとアイテムをそれぞれ独立した「タワー」と呼ばれるニューラルネットワークでエンベディングするモデル。
- ユーザータワーはユーザーIDを入力とし、アイテムタワーはアイテムIDを入力とします。
- 各タワーはIDを埋め込み、複数の線形層（LinearBlock）を通して最終的なユーザー/アイテム表現ベクトルを出力します。
- 訓練時には、ユーザーベクトルと正例アイテムベクトルとの類似度（内積）が高く、負例アイテムベクトルとの類似度が低くなるように学習します (BCEWithLogitsLossを使用)。
- 評価時には、ユーザーベクトルと候補アイテムベクトルの類似度を計算し、ランキング上位のアイテムを推薦します。

### SASRec

- **ファイルパス:** `src/models/sasrec.py`
- TransformerのSelf-Attention機構を利用したシーケンシャル推薦モデル。
- ユーザーの過去のアイテムインタラクション履歴（シーケンス）を入力とします。
- アイテムIDを埋め込み、位置エンコーディングを加えた後、複数のTransformerエンコーダーブロック（自己注意機構 + FeedForward層）で処理します。
- 自己注意機構により、シーケンス内のアイテム間の依存関係を捉えます。
- 最後のTransformerブロックの出力（特にシーケンスの最後のアイテムに対応する表現）を用いて、次にユーザーがインタラクションするアイテムを予測します。
- 訓練時には、予測アイテム（正例）と負例アイテムに対するスコアを計算し、正例のスコアが高くなるように学習します (BCEWithLogitsLossを使用)。
