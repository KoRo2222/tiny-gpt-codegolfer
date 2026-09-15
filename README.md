# tiny-gpt-codegolfer

BPEトークナイザー→小型GPT-2→事前学習→強化学習→SFTという流れで、ゼロから作るコードゴルフ特化のLLM。短くてテストに通るPythonコードを書けるように育てていく個人プロジェクト。

## 進捗

- [x] BPEトークナイザー(`src/codebot/tokenizer`)
  - 学習アルゴリズム(バイトレベルBPE、GPT-2方式)
  - `<|endoftext|>`特殊トークンと、複数文書をEOT区切りの1本のid列に詰める`encode_with_eot`
- [x] 事前トークン化(`src/codebot/data`) — コーパスをあらかじめid列に変換し、`data/train.bin` / `data/val.bin`として保存(学習ループが毎回テキストを読まず`np.memmap`で読める形式)
- [x] 小型GPT-2の実装(`src/codebot/model`) — 1ブロック分のTransformerデコーダー
  - Attention — 「ソフトなディクショナリ」としてのscaled dot-product attention。query-key類似度→softmax→valueの重み付き和。causalマスク対応
  - `TokenPositionalEmbedding` — トークン埋め込み+位置埋め込み
  - `MultiHeadAttention` — Attentionをヘッドに分けて並列適用
  - `FeedForward` — 位置ごとのLinear→GELU→Linear
  - `TinyGPT` — Embed→Attention(残差)→FFN(残差)→Linear(lm_head)→(softmaxは`next_token_probs`で分離)
- [ ] 事前学習
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

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
