# tiny-gpt-codegolfer

BPEトークナイザー→小型GPT-2→事前学習→SFT→強化学習という流れで、ゼロから作るコードゴルフ特化のLLM。短くてテストに通るPythonコードを書けるように育てていく個人プロジェクト。

## 進捗

- [x] BPEトークナイザー(`src/codebot/tokenizer`)
  - 学習アルゴリズム(バイトレベルBPE、GPT-2方式)
  - `<|endoftext|>`特殊トークンと、複数文書をEOT区切りの1本のid列に詰める`encode_with_eot`
- [x] 事前トークン化(`src/codebot/data`) — コーパスをあらかじめid列に変換し、`data/train.bin` / `data/val.bin`として保存(学習ループが毎回テキストを読まず`np.memmap`で読める形式)
- [x] データセット拡充(`src/codebot/data/task_catalog.py`) — 数値/文字列/リスト/探索・ソート/クラスにわたる関数・クラス75件を1つのカタログとして定義。`scripts/build_datasets.py`がここから事前学習コーパス・SFT例・RLタスク(テスト付き)を自動生成するので、3つのデータセットが食い違わない。生成時にRLタスクのテストを実際に実行して自己検証(不正なテストは即エラー)
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
  - `scripts/pretrain.py` — 学習前後の生成テキストを見比べられるCLI。チェックポイントはモデル構成(config)も一緒に保存(`save_checkpoint`/`load_checkpoint`)。コーパス拡充後も2500トークン程度と小さく、なお過学習気味(train_loss/val_lossの乖離)だが、モデル・データともまだ小さい個人プロジェクトの範囲では想定通り
  - `scripts/generate.py` — 学習済みチェックポイントから任意のプロンプトでテキスト生成するCLI(`--temperature`, `--seed`指定可)
- [x] SFT(`src/codebot/train/sft.py`) — 指示文(プロンプト)→コード(応答)のペアで事前学習済みモデルを微調整
  - `build_example` — プロンプト+応答+EOTを1本のid列にし、損失は応答部分のトークンだけに掛ける(プロンプト部分は`ignore_index`でマスク)
  - `sft_loop` — 1件ずつ(batch_size=1)optimizer stepするシンプルな実装。paddingやattentionマスクをまだ持っていないためバッチ化は見送り
  - `data/sft/examples.jsonl` — 「〜する関数を書け」という指示文と、対応するPython関数/クラスのペア75件(`task_catalog.py`から自動生成)
  - `scripts/sft.py` — 事前学習済みチェックポイントを読み込んでSFTし、`data/checkpoint_sft.pt`に保存。学習前後の生成を比較可能
- [x] 強化学習(`src/codebot/rl`) — GRPO(Group Relative Policy Optimization)、報酬はコード実行結果(テスト通過+短いほど高得点)
  - `reward.code_golf_reward` — 生成コードをsubprocessで実行(タイムアウト付き、無限ループ対策)しテストを通すか判定。通れば1.0+短さボーナス、通らなければ0.0
  - `sampling.sample_completion` — 1本の補完をサンプリングしつつ、各トークンのサンプリング時log-probを記録(GRPOの重要度比の分母)
  - `grpo.group_relative_advantages` — 同じプロンプトから採った複数サンプル(グループ)内で報酬を正規化。PPOの学習済み価値関数の代わりにこれをベースラインにするのがGRPOの要点
  - `grpo.grpo_loss` — クリップ付き重要度比サロゲート損失+(参照モデルに対する)KL正則化項。全員同じ報酬(学習シグナルなし)のグループはスキップ
  - `data/rl/tasks.jsonl` — テストアサーション付きのコードゴルフお題73件(`task_catalog.py`から自動生成)
  - `scripts/rl.py` — SFTチェックポイントを起点に(同じ重みを凍結した参照モデルとしても使用)GRPOで学習、`data/checkpoint_rl.pt`に保存
  - 「アドバンテージが正の時に実際にそのレスポンスの確率が上がる」ことをユニットテストで直接確認済み
  - `reward.run_tests`は生成コードを一時ファイルに書いてから実行する方式。以前は`python -c <script>`の引数文字列として渡していたため、バイトレベルBPEがヌル文字(`\x00`)を含むトークン列をサンプルした瞬間に`subprocess.run`が`ValueError`で丸ごとクラッシュするバグがあった(300stepの実ランで実際に踏んだ)
- [x] 評価スクリプト(`src/codebot/rl/evaluate.py`) — 学習の効果を1枚のサンプルの目視ではなく数値で追えるように
  - `evaluate_model` — 全RLタスクに対してpass@num_samplesと(通過した場合の)平均コード長を集計。`sample_fn`を差し替え可能にしてあり、テストは本物のモデルではなく固定の偽サンプラーで採点ロジックだけを検証
  - `scripts/evaluate.py` — タスクごとのpass/fail一覧+「solved N/M」を表示し、`data/eval_history.jsonl`に追記(実行のたびに記録が積み上がるので、学習を伸ばした効果を後から比較できる)
  - データ拡充後にGRPOを300step(group_size=12)まで伸ばし、`--max-new-tokens`を60→120に上げて(ソート系アルゴリズムが生成途中で打ち切られて不当に失格していたバグを修正)評価し直したところ、**73タスク全て通過**(pass@10, temperature=1.0)。`caesar_cipher`/`rot13`/ソート3種/`binary_search`は依然としてpass_rateが0.1〜0.5程度と低い(=たまにしか解けない)ものの、以前は0%だったところから改善した

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
