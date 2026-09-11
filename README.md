# tiny-gpt-codegolfer

Building a small code-golf-playing LLM from scratch: BPE tokenizer → tiny
GPT-2 → pretraining → RL → SFT.

## Progress

- [x] BPE tokenizer (`src/codebot/tokenizer`)
- [ ] Tiny GPT-2 implementation
- [ ] Pretraining
- [ ] RL (execution-based reward: tests pass + shorter code wins)
- [ ] SFT

## Tokenizer

```
python -m venv .venv
.venv/Scripts/pip install pytest

# train on data/corpus and save to data/tokenizer.json
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 512

# run tests
.venv/Scripts/python -m pytest tests/ -v
```
