"""Measurement surface: judge verdicts, retrieval metrics, run artifacts, reports.

Everything in here answers "did retrieval help?" without asking an answer model
anything. The oracle is a judge model grading candidate chunks (ADR-0008); its
verdicts are cached by `(question_id, chunk_id, rubric sha)`, so a new strategy
costs judge calls only for the chunks it surfaced that nobody has seen before,
and every metric is arithmetic over that cache. That asymmetry is the reason
retrieval can be iterated on for free offline when an answer-accuracy run of the
same sample is hours of generation.

Modules:

* `metrics` -- recall@k, relevant_fraction@k, first-relevant-rank. Pure.
* `rubric` -- the judging prompt and its sha. Editing it invalidates every
  cached verdict on purpose: a verdict is a fact about a chunk *under one rubric*.
* `judge` -- the verdict cache and the only code that spends endpoint time.
* `store` -- the `chunks.jsonl` sidecar that lets a run be re-read without the
  380k-row index.
* `report` -- one grid renderer behind the terminal text, `report.html`, and
  `FINDINGS.md`.
* `runlog` -- pinned samples, the row schema, the run-dir writer, the endpoint
  circuit breaker. (Formerly `scripts/exploration/_common.py`.)
* `harness` / `inspector` / `lab` -- the grid runner, the offline run reader, and
  the localhost knob panel. Library modules with CLI entry points; `scripts/`
  holds the thin commands that call them.
* `rewrite` -- query reformulation, kept importable but unscheduled: it is the
  one lever that still costs completions.
"""
