# AI_exercise
VLMに間違い探しを学習させるプロジェクト

### セットアップ
グローバルにuvを追加します
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

最初に仮想環境を作成してライブラリをインストールします。
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

次に、.venv/lib/python/site-packages/unsloth_zoo/vllm_utils.pyの中で、limit_mm_per_promptを2に変更します。
```python
"limit_mm_per_prompt": {"image": 2, "video": 0}
```

### 使い方
configs/default.yamlを参考に学習パラメータを設定したYAMLファイルを作成してください。
```yaml
model_name: "unsloth/Qwen2.5-VL-7B-Instruct"
max_seq_length: 32768
load_in_4bit: true
fast_inference: true
gpu_memory_utilization: 0.8
...
```

そして、以下のように実行できます。
```bash
uv run trainer.py --config (YAMLファイルのパス)
```