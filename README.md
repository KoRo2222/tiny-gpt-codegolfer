# tiny-gpt-codegolfer

BPEトークナイザー→小型GPT-2→事前学習→強化学習→SFTという流れで、ゼロから作るコードゴルフ特化のLLM。短くてテストに通るPythonコードを書けるように育てていく個人プロジェクト。

## 進捗

- [x] BPEトークナイザー(`src/codebot/tokenizer`)
  - 学習アルゴリズム(バイトレベルBPE、GPT-2方式)
  - `<|endoftext|>`特殊トークンと、複数文書をEOT区切りの1本のid列に詰める`encode_with_eot`
- [x] 事前トークン化(`src/codebot/data`) — コーパスをあらかじめid列に変換し、`data/train.bin` / `data/val.bin`として保存(学習ループが毎回テキストを読まず`np.memmap`で読める形式)
- [ ] 小型GPT-2の実装
- [ ] 事前学習
- [ ] 強化学習(報酬: テスト通過 + コードが短いほど高得点)
- [ ] SFT

## トークナイザー

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# data/corpus 以下のコーパスで学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 512

# コーパスを事前トークン化し、data/train.bin, data/val.bin に保存
.venv/Scripts/python scripts/prepare_data.py

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
