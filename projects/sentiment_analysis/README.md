# Sentiment Analysis

IMDB の映画レビューを使って、Transformer Encoder をフルスクラッチ実装で学習・評価するための project です。  
テキスト分類の基本要素である前処理、語彙構築、データセット実装、Transformer Encoder、本体の分類ヘッドまでを一通り含んでいます。

## 概要

この project では、映画レビュー文を入力として、レビューの感情を **positive / negative** の 2 値分類で予測します。

- データセット: [IMDb Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- タスク: binary sentiment classification
- 実装方針:
  - PyTorch / Lightning ベース
  - Transformer Encoder を自前実装
  - `uv` と `Makefile` を使った開発フロー

## データセット

学習には `datasets` ライブラリ経由で取得する IMDb データセットを利用します。

- train: 25,000 件
- test: 25,000 件
- unsupervised: 50,000 件
- ラベル:
  - `0`: negative
  - `1`: positive

ランダム予測のベースライン精度はおよそ 50% です。

参照:

- <https://huggingface.co/datasets/stanfordnlp/imdb>

## 実装内容

この project には主に以下が含まれます。

- `data/`
  - IMDb データセットの取得
  - テキストの tokenization / normalization
  - 語彙構築
  - fixed length の系列への変換
  - Lightning DataModule
- `models/`
  - Transformer Encoder
  - Self-Attention
  - Embedding
  - Feed Forward Network
  - Sequence classification head
- `tests/`
  - モジュールごとのテスト

## ディレクトリ構成

```text
sentiment_analysis/
├── Makefile                 # 開発用コマンド
├── README.md                # このファイル
├── pyproject.toml           # Python package 設定
├── train.py                 # 学習エントリポイント
├── uv.lock                  # lock file
├── data/                    # データ取得・前処理・DataModule
│   └── dataset.py
├── img/                     # README 用画像など
├── models/                  # モデル実装
│   ├── classifier.py
│   └── modules/
│       ├── decoder.py
│       ├── encoder.py
│       └── base/
├── notebook/                # 実験・確認用 notebook
├── tests/                   # テスト
└── utils/                   # 補助ユーティリティ
```

## セットアップ

package root で作業してください。

```sh
cd projects/sentiment_analysis
make install
```

`make install` により、`uv` を通して依存関係をセットアップします。

## 学習

基本的な学習実行は以下です。

```sh
cd projects/sentiment_analysis
make train
```

環境によっては、`Makefile` の定義に従って `uv run ...` ベースで学習が実行されます。

## テスト

テストは package root で実行します。

```sh
cd projects/sentiment_analysis
make test
```

## Lint / Format

コード品質確認と整形も package root で行います。

```sh
cd projects/sentiment_analysis
make lint
make fmt
```

## モデル

分類モデル本体は `models/classifier.py` にあり、主に以下の構成です。

- `TransformerForSequenceClassification`
  - Transformer Encoder を用いて入力系列をエンコード
  - 最終表現を分類ヘッドに渡して 2 値分類を実施
- `Classifier`
  - dropout + linear を中心としたシンプルな分類ヘッド

また、`models/modules/` 以下に Self-Attention、Embedding、Encoder / Decoder block などの基礎モジュールがあります。  
主用途は sentiment classification ですが、Transformer の構成要素を理解しやすいように分割されています。

## 前処理

`data/dataset.py` では主に以下を行います。

- IMDb データセットの取得
- spaCy を用いた text normalization / tokenization
- vocabulary 構築
- padding を含む固定長系列への変換
- `IMDbDataset` / `IMDbDataModule` の提供

このため、モデル学習だけでなく、テキスト分類パイプライン全体の実験用 project として扱えます。

## 補足

この project は他の推薦系 project と比べると独立性が高く、`libs/ml_sandbox_libs` への依存を前提としない構成です。  
一方で、monorepo 内の他 package と同様に `uv` と `Makefile` ベースの開発フローに揃えると扱いやすくなります。

## 参考

- IMDb Dataset: <https://huggingface.co/datasets/stanfordnlp/imdb>
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 比較参考: [DistilBERT](https://arxiv.org/abs/1910.01108)