---
name: python-project-workflow
description: "Use when: このリポジトリの Python package で作業し、package root での `make install` / `make fmt` / `make lint` / `make test` の順序、`uv` 前提の実行方法、Google style docstring、`ruff` / `mypy` / `pytest` failure の直し方を確認したい。"
---

# Python Project Workflow

この skill は、このリポジトリ内の Python package に共通する作業手順を確認するときに使います。

model 設計や shared module 切り出しが主題なら、`recsys-model-workflow` または `shared-module-extraction` を優先します。

## 対象

- `libs/ml_sandbox_libs`
- `projects/recsys-ranking`
- `projects/recsys-candidate-generation`
- `apps/vertex-job-runner`
- `pyproject.toml` と `Makefile` を持つ他の Python package

## 基本ルール

- Python 実行は `uv` を通す
- まず package root へ移動する
- CI は package ごとに `make install`, `make lint`, `make test` を実行する
- test は `pytest` を使い、直接叩かず `make test` または `uv run pytest` で実行する
- mock が必要な test では `pytest-mock` の `mocker` fixture を使い、`unittest.mock` を直接 import しない
- Python の docstring は既存コードに合わせて Google style を使う
- 関数・method の docstring には、処理内容、`Args`, `Returns`, `Raises` を明記する

## Docstring Style

- style は Google style を採用し、見出しは `Args:`, `Returns:`, `Raises:` を使う
- 先頭の要約で「その関数が何をするか」を 1-2 文で書く
- 大きい model class では、要約に加えて `Architecture` や `Key characteristics` を書いてよい
- utility や単純な helper では、過剰に長い背景説明を書かず簡潔に保つ
- `Args` には各引数の意味、前提条件、必要なら単位や shape を書く
- `Returns` には返り値の意味を書く
- `Raises` には入力不正、内部バリデーション失敗、依存ライブラリ由来を含む主要な例外条件を書く
- Python では型注釈を参照できるため、docstring 内で引数や返り値の型を重ねて書かなくてよい
- `Example` セクションは原則不要で、読みやすさを優先する
- 実装を変えたのに docstring が古くなる状態を残さない

## 標準手順

1. package root を確認する。
2. 必要なら `make install` を実行する。
3. code を修正したなら `make fmt` を実行する。
4. `make lint` を実行する。
5. `make test` を実行する。

## 失敗時の進め方

- `ruff` error は対象 file のみを直す
- `mypy` error は `type: ignore` の追加ではなく、まず型の整合を直す
- test failure は root cause を直す
- unrelated failure はむやみに触らず、変更と関係する範囲を優先する
- test 追加や修正で mock が必要なら、既存 code を含めて `pytest-mock` へ寄せる

## よくある補足

- `apps/vertex-job-runner` は README にある通り `uv sync` と `uv run pytest` でも確認できる
- `libs/ml_sandbox_libs` を変更した場合は、library 単体だけでなく影響先 project も追加で確認する
- import path を動かしたら lint と test の両方を回す

## 完了条件

- 少なくとも変更した package の `make lint` と `make test` が通る
- `libs` を変更した場合は必要な downstream project も確認できている
- CI で落ちると分かる状態を残していない
