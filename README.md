# AI_exercise

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