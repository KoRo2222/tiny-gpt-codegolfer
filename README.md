# tiny-gpt-codegolfer

BPEトークナイザー→小型GPT-2→事前学習→強化学習→SFTという流れで、ゼロから作るコードゴルフ特化のLLM。短くてテストに通るPythonコードを書けるように育てていく個人プロジェクト。

## 進捗

- [x] BPEトークナイザー(`src/codebot/tokenizer`)
  - 学習アルゴリズム(バイトレベルBPE、GPT-2方式)
  - `<|endoftext|>`特殊トークンと、複数文書をEOT区切りの1本のid列に詰める`encode_with_eot`
- [x] 事前トークン化(`src/codebot/data`) — コーパスをあらかじめid列に変換し、`data/train.bin` / `data/val.bin`として保存(学習ループが毎回テキストを読まず`np.memmap`で読める形式)
- [x] 小型GPT-2の実装(`src/codebot/model`) — GPT-2方式のTransformerデコーダー
  - Attention — 「ソフトなディクショナリ」としてのscaled dot-product attention。query-key類似度→softmax→valueの重み付き和。causalマスク・1/√d_kスケーリング対応
  - `TokenPositionalEmbedding` — トークン埋め込み+位置埋め込み(絶対位置埋め込み)
  - `MultiHeadAttention` — Attentionをヘッドに分けて並列適用。QKV射影とValue行列を使用
  - `FeedForward` — 位置ごとのLinear→GELU→Linear
  - `TransformerBlock` — Attention(Pre-LN, 残差)→FFN(Pre-LN, 残差)
  - `TinyGPT` — Embed→`TransformerBlock`をn_layers層スタック→最終LayerNorm(ln_f)→Linear(lm_head、埋め込み層と重み共有)→(softmaxは`next_token_probs`で分離)
- [x] 事前学習(`src/codebot/train`) — `train.bin`からランダムな窓を切り出してnext-token予測、cross entropy loss、AdamWで学習
  - `get_batch` — (入力, 1つずらしたターゲット)のペアをランダムサンプリング
  - `train_loop` — 学習ステップ+定期的にval lossを評価。`TinyGPT.generate`で温度付きサンプリング生成も可能に
  - `scripts/pretrain.py` — 学習前後の生成テキストを見比べられるCLI。チェックポイントはモデル構成(config)も一緒に保存(`save_checkpoint`/`load_checkpoint`)。現状のコーパスは899トークンしかなく過学習気味(train_loss/val_lossの乖離)だが、`if`/`return`など実際のコードらしいトークンが出るようになることは確認済み
  - `scripts/generate.py` — 学習済みチェックポイントから任意のプロンプトでテキスト生成するCLI(`--temperature`, `--seed`指定可)
- [ ] 強化学習(報酬: テスト通過 + コードが短いほど高得点)
- [ ] SFT

## セットアップ

```
python -m venv .venv
# CPU版torchを明示的に入れる(付けないとCUDA同梱の巨大なwheelが入る)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy pytest

# data/corpus 以下のコーパスで学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 512

# コーパスを事前トークン化し、data/train.bin, data/val.bin に保存
.venv/Scripts/python scripts/prepare_data.py

# 事前学習(data/checkpoint.pt に保存)
.venv/Scripts/python scripts/pretrain.py --steps 500

# 学習済みチェックポイントからテキスト生成
.venv/Scripts/python scripts/generate.py --prompt "def " --temperature 0.8

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
