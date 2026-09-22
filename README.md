# Problem-AI-KCN

Local RAG system for industrial document retrieval using BM25, semantic search, and Reciprocal Rank Fusion (RRF).

## Overview

This project is a local Retrieval-Augmented Generation (RAG) retrieval system for experimenting with hybrid search.

The retrieval pipeline combines:

- BM25 keyword search
- Local embedding search
- Cosine similarity
- Reciprocal Rank Fusion (RRF)
- Local GGUF embedding model
- `llama-cpp-python`

The current project focuses on the retrieval stage of RAG.

```text
                         Query
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        BM25 Search               Embedding Search
             │                           │
             ▼                           ▼
       Ranked Results              Ranked Results
             │                           │
             └─────────────┬─────────────┘
                           ▼
                          RRF
                           │
                           ▼
                    Final Ranked Results
```

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

### 3. Activate virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\activate
```

#### Windows Git Bash

```bash
source .venv/Scripts/activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Download Embedding Model

This project uses the following local embedding model:

```text
jina-embeddings-v5-text-nano-retrieval
```

GGUF model:

```text
v5-nano-retrieval-Q4_K_M.gguf
```

The model is **not included in this repository** because its file size exceeds GitHub's 100 MB file size limit.

Download the model from Hugging Face:

https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/blob/main/v5-nano-retrieval-Q4_K_M.gguf

After downloading, create the `models` directory:

```bash
mkdir models
```

Place the downloaded model inside:

```text
models/
└── v5-nano-retrieval-Q4_K_M.gguf
```

The application expects the model at:

```text
models/v5-nano-retrieval-Q4_K_M.gguf
```

## Project Structure

```text
Problem-AI-KCN/
│
├── models/
│   └── v5-nano-retrieval-Q4_K_M.gguf
│
├── data/
│   └── embeddings.json
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The GGUF model is intentionally excluded from Git:

```gitignore
models/*.gguf
```

## Run the Project

After installing the dependencies and downloading the embedding model:

```bash
python main.py
```
