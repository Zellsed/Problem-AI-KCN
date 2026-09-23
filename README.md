# Problem-AI-KCN

Local RAG system for industrial document retrieval using BM25, semantic search, and Reciprocal Rank Fusion (RRF).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Zellsed/Problem-AI-KCN.git
cd Problem-AI-KCN
```

### 2. Create virtual environment

Using `uv`:

```bash
uv venv
```

## Download Embedding Model

The model is **not included in this repository** because its file size exceeds GitHub's 100 MB file size limit.

Download the model from Hugging Face:

https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/blob/main/v5-nano-retrieval-Q4_K_M.gguf

After downloading, create the `models` directory:

```bash
mkdir models
```

Place the downloaded model inside:

```text
models
```

```text
embedding/
└── v5-nano-retrieval-Q4_K_M.gguf
```

```text
llm/
└── Qwen3-4B-Q4_K_M.gguf
```
