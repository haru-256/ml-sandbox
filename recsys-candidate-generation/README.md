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

## モデル

具体的には以下のモデルを実装する予定です。

- [x] TwoTower: Two-Tower Model
- [ ] MF: Matrix Factorization
- [ ] Collaborative Filtering
- [ ] NCF: Neural Collaborative Filtering
- [ ] NeuMF: Neural Matrix Factorization
- [ ] NGCF: Neural Graph Collaborative Filtering
- [ ] LightGCN
- [ ] GRU4Rec: Gated Recurrent Unit for Sequential Recommendation
- [ ] SASRec: Self-Attentive Sequential Recommendation
- [ ] BERT4Rec: BERT for Sequential Recommendation
- [ ] gSASRec
