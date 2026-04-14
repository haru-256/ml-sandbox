---
name: shared-module-extraction
description: "Use when: `projects/recsys-ranking` と `projects/recsys-candidate-generation` など複数 project にある class / function / dataclass / enum の重複を `libs/ml_sandbox_libs` へ共通化し、shared module への移動、import path の整理、`libs` 側 unit test と project 側 integration test の再配置まで進めたい。"
---

# Shared Module Extraction

この skill は、project 内の重複実装を `libs/ml_sandbox_libs` へ寄せるときに使います。

Python package 共通の lint / test / docstring ルール確認が必要なら、`python-project-workflow` も併用します。

## 目的

- duplicate を減らす
- shared code を `libs/ml_sandbox_libs` に集約する
- project 固有 logic と shared logic の境界を明確にする
- shared module の単体 test を `libs` 側へ寄せる

## 前提

- Python 関連の実行は `uv` を通す
- 変更は最小限に留める
- 互換レイヤを増やすより、可読性と責務分離を優先する

## 作業手順

1. まず shared 候補を確認する。
    - 複数 project に同じ class / function / dataclass / enum が存在するかを見る。
    - project 固有の business logic まで `libs` に持ち込まない。

2. 置き場所を決める。
    - model 用の shared 型は `ml_sandbox_libs.models.types`
    - optimizer 用の shared 型は `ml_sandbox_libs.optimizer.types`
    - 再利用される model building block は `ml_sandbox_libs.models.modules`

3. `libs` に実装を追加する。
    - public API は必要以上に複雑にしない。
    - project 側 wrapper を増やさず、可能なら `libs` を直接使う形にする。
    - public function / method を追加または実質変更したら docstring も更新する。

4. project 側 import を差し替える。
    - old module を残すより、参照元を直接更新する。
    - ただし project 固有の wiring は project 側に残す。

5. test を整理する。
    - shared module の単体 test は `libs/ml_sandbox_libs/tests` に置く。
    - project 側には integration test を残す。
    - shared 実装だけを確認する重複 test は project 側から削る。

## 完了条件

- `libs` に shared code が集約されている
- project 側が `libs` を直接 import している
- shared module の単体 test が `libs` 側にある
- 重複 test が整理されている
- 変更した package の `make lint` と `make test` が通る

## 注意点

- formatting や import 並び替えだけで unrelated file を広く触らない
- public な import path を変えたら README も更新する
- `libs/ml_sandbox_libs` を変更したら、影響先 project も確認する
