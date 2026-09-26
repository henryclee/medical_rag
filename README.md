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

Configure the served LLM endpoints first. `config/models.yaml` names the
endpoint, model id, sampling parameters, and HTTP timeout/retries for
`model_a`/`model_b`; the API keys are read from the environment variable each
entry names, so they live in a gitignored `.env` at the repo root (copy
`.env.example` and fill it in):

```bash
set -a; source .env; set +a     # exports MODEL_A_API_KEY / MODEL_B_API_KEY
```

Then:

```bash
# Build the retrieval index (idempotent; downloads StatPearls on first run).
.venv/bin/python scripts/build_index.py --corpus statpearls

# Run the tests. Everything is offline except the single `live` test, which
# skips unless MEDICAL_RAG_LIVE_TESTS=1 is set.
.venv/bin/pytest
MEDICAL_RAG_LIVE_TESTS=1 .venv/bin/pytest   # also hits the real endpoint

# TODO: run_pilot.py (dev split) and run_experiment.py (test split) are not
# implemented yet -- see PLAN.md, Phases 5-13.
```
