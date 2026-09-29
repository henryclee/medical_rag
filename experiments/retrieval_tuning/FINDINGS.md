# Retrieval tuning -- judge harness FINDINGS

Not yet run against the full pinned 20-question sample.

`judge_harness.py` has been built and smoke-tested (`--dry-run`, then a live
1-question run at `outputs/exploration/retrieval_tuning/20260928T210347Z`,
then `--resume` on the same run dir to confirm it skips already-judged
questions rather than re-firing calls). The live smoke test produced
correctly-shaped, gold-blind judgments (e.g. the gold-supporting chunk for
q1312 was labelled `relevant, supports_options=["B"]` without ever being
told the answer was B) across all six methods (`dense`, `dense_rerank`,
`bm25`, `hybrid`, `hybrid_rerank`, `reform_dense`).

This file is overwritten by the harness itself on every run
(`--findings-path`, default this path) -- run it against the full sample to
get the real comparison table:

```
set -a; source .env; set +a
.venv/bin/python experiments/retrieval_tuning/judge_harness.py
```
