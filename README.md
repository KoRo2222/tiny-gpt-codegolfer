# tiny-gpt-codegolfer

BPEトークナイザー、GPT-2アーキテクチャ、事前学習、SFT、強化学習(GRPO)まで、全てゼロから自前実装した小型LLM。短くてテストに通るPythonコードを書くこと(コードゴルフ)に特化して育てている。

## 構成

- **トークナイザー**(`src/codebot/tokenizer`) — バイトレベルBPEを学習アルゴリズムから自前実装。GPT-2方式のプリトークナイズ、`<|endoftext|>`による文書区切り
- **モデル**(`src/codebot/model`) — GPT-2方式のTransformerデコーダー。Multi-Head Attention、位置ごとのFeedForward、Pre-LN残差ブロック、埋め込み層とLM headの重み共有
- **データ**(`src/codebot/data/task_catalog.py`) — 数値・文字列・リスト・探索/ソート・クラスにまたがる75件の関数/クラスを1つのカタログとして定義し、事前学習コーパス・SFT例・RLタスク(テスト付き)をそこから自動生成。3つのデータセットが食い違わない
- **学習**(`src/codebot/train`) — next-token予測による事前学習と、プロンプト部分をマスクしたSFT(指示追従)
- **強化学習**(`src/codebot/rl`) — GRPO。プロンプトごとに複数の補完をサンプリングし、実際にコードを実行してテスト通過+短さを報酬に、クリップ付き重要度比サロゲート損失とKL正則化で方策を更新
- **評価**(`src/codebot/rl/evaluate.py`) — 全タスクに対するpass rateを記録し、学習の効果を数値で追跡

## 現状

コードゴルフお題73件全てで(少なくとも稀には)正解を出せる。四則演算・文字列判定・リスト操作などの単純なお題はほぼ確実に解けるが、ソートアルゴリズムや暗号化のような複数行アルゴリズムはまだ成功率が低い。パラメータ数・データ量ともに小さい個人プロジェクトの規模なので、未知のお題への汎化はまだ弱く、既知のパターンの組み合わせが中心。

## セットアップ

```
python -m venv .venv
# CPU版torchを明示的に入れる(付けないとCUDA同梱の巨大なwheelが入る)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy pytest

# task_catalog.py から corpus/SFT例/RLタスクを生成(テストを自己検証)
.venv/Scripts/python scripts/build_datasets.py

# data/corpus 以下のコーパスで学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 1024

# コーパスを事前トークン化し、data/train.bin, data/val.bin に保存
.venv/Scripts/python scripts/prepare_data.py

# 事前学習(data/checkpoint.pt に保存)
.venv/Scripts/python scripts/pretrain.py --steps 500

# 学習済みチェックポイントからテキスト生成
.venv/Scripts/python scripts/generate.py --prompt "def " --temperature 0.8

# SFT(data/sft/examples.jsonl で指示追従を学習、data/checkpoint_sft.pt に保存)
.venv/Scripts/python scripts/sft.py --epochs 20

# GRPO(data/rl/tasks.jsonl のテストを通すよう強化学習、data/checkpoint_rl.pt に保存)
.venv/Scripts/python scripts/rl.py --steps 100

# 全RLタスクに対するpass rateを評価(data/eval_history.jsonl に記録)
.venv/Scripts/python scripts/evaluate.py --checkpoint data/checkpoint_rl.pt

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
