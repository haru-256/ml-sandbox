---
name: recsys-model-workflow
description: "Use when: `projects/recsys-ranking` や `projects/recsys-candidate-generation` の推薦 model を変更し、tower / DIN / DLRM / DeepFM / SASRec などの構成変更、project 固有実装と `libs/ml_sandbox_libs` に寄せる shared building block の切り分け、model 周りの test 配置や責務分離を判断したい。"
---

# Recsys Model Workflow

この skill は、推薦 model 周りの変更を進めるときに使います。

重複実装を `libs` へ移す作業自体が主題なら、`shared-module-extraction` も併用します。

## 対象 project

- `projects/recsys-ranking`
- `projects/recsys-candidate-generation`

## 判断基準

- model 固有の composition や training flow は project 側に残す
- embedding, MLP, attention, pooling, enum, dataclass のような building block は `libs` への共通化を検討する
- shared 型は責務ごとに分ける

## 既存の方針

- model 用の shared 型は `ml_sandbox_libs.models.types`
- optimizer 用の shared 型は `ml_sandbox_libs.optimizer.types`
- shared module は `ml_sandbox_libs.models.modules`
- project 側では shared module への wrapper を増やしすぎない

## 実装時のチェックポイント

1. 変更が project 固有か shared 候補かを先に切り分ける。
2. 文字列設定を受ける factory 層では、必要に応じて enum へ変換してから model に渡す。
3. model 本体の constructor は、できるだけ型付きの enum や dataclass を受ける。
4. shared module を使うなら、project 側の重複実装や重複 test を整理する。
5. `libs` を触る場合は、library 単体だけでなく downstream project の確認も前提にする。

## test 方針

- 意味のない test は入れない。実装を変えても利用者価値や回帰検知につながらない確認は避ける
- 保守性が低い test は入れない。内部実装の細部、偶然の出力、過剰な fixture に依存する test は避ける
- test は仕様、契約、回帰しやすい振る舞いを確認するために書く
- 1つの test では 1つの責務を確認し、失敗時に原因を追いやすくする
- 同じ振る舞いを複数箇所で重複確認しない
- 入出力、主要なバリデーション、公開 API の振る舞いのように、変更時の影響が大きい箇所を優先して test する
- mock が必要なときは `pytest-mock` の `mocker` fixture を使い、`unittest.mock` の直 import は増やさない

## 推奨確認コマンド

- `projects/recsys-ranking`: `make fmt && make lint && make test`
- `projects/recsys-candidate-generation`: `make fmt && make lint && make test`
- `libs/ml_sandbox_libs`: `make fmt && make lint && make test`

## 完了条件

- code の責務分離が明確
- shared 化すべきものが `libs` に集約されている
- project 側に残すべき integration test が残っている
- 関連 package の lint / test が通っている
