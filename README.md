# harus-ai

## open-webui

prequisite: ollama

```sh
uv venv python 3.11
uv python pin 3.11

uv run open-webui serve
```

## langchain with huggingface

```bash
uv add langchain langchain-community langchain-huggingface
uv add transformers torch accelerate bitsandbytes
uv add sentence-transformers
uv add python-dotenv  # for environment variables
```
