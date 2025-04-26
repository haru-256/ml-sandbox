# ML-Sandbox

[![Python Lint and Test](https://github.com/haru-256/ml-sandbox/actions/workflows/python-lint-test.yml/badge.svg)](https://github.com/haru-256/ml-sandbox/actions/workflows/python-lint-test.yml)

機械学習モデルの実装と実験用のリポジトリ

## ディレクトリ構成

```sh
.
├── README.md
├── apps/  # 機械学習コードを動かすアプリケーションコード
│   └── job_runner
├── libs/ # 複数projectで使用される内部ライブラリ
│   └── ml_sandbox_libs
└── projects/ # 各問題設定に対応するproject
    └── recsys-candidate-generation
```

## projects

### Recsys Candidate Generation

Candidate Generationとは、以下2段階の推薦のMulti-Stage Architectureの1つ目の段階を指します。
どれだけユーザーが興味のある商品を絞り込むことができるかが重要です。

1. Candidate Generation: ユーザーが興味のあるitemを取得し推薦候補を生成
2. Ranking: 取得した推薦候補を並び替える

詳細は[README.md](projects/recsys-candidate-generation/README.md)を参照してください。

## apps

### job_runner

Vertex AIでのトレーニングジョブを実行するためのCLIツール

## libs

### ml_sandbox_libs

共通ライブラリ。主に以下の機能を提供します。

- Amazon Reviews 2023のデータセットを扱うためのユーティリティ
