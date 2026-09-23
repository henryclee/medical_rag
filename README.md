# Medical RAG Experiment

## Overview

A research pipeline for evaluating retrieval-augmented generation (RAG)
strategies on medical question-answering tasks (MedQA, StatPearls corpus).
The LLM is served by a separate OpenAI-compatible endpoint and is never
loaded in-process; this repo only handles retrieval, orchestration, and
analysis.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

TODO: fill in once the pipeline is implemented (build index, run pilot,
run full experiment).
