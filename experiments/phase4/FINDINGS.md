# Phase 4 — Generation client + prompts findings

Exploratory/implementation evidence. Not a result (see `../README.md`). Phase 4
predates the section 5 artifact convention; this file applies it retroactively
to the measurements that drove Phase 4's implementation decisions, so PLAN.md
can state the decisions without re-carrying the raw numbers.

- **Command:** live probes via `.venv/bin/python scripts/probe_models.py` and
  ad hoc `LLMClient` calls against the two local `mlx_lm.server` endpoints.
- **Endpoints:** `model_a` = Qwen2.5-7B-Instruct-8bit on `localhost:8081`;
  `model_b` = DeepSeek-R1-Distill-Qwen-7B-8bit on `localhost:8082`.

## 1. Question this phase asked

Does `LLMClient` work end-to-end against the real endpoints (plain generation,
structured output, recovery from a truncated response), and are `model_a` /
`model_b` genuinely different enough (capability, latency, failure mode) to
serve as the study's two-model axis?

## 2. Live verification

- 3 closed-book + 3 retrieval-augmented MedQA-style questions, both models:
  all returned `finish_reason="stop"` and parsed to valid letters (closed-book
  A/B/B, RAG A/B/B — all three RAG answers medically correct).
- `agenerate_structured()` round-tripped both JSON shapes the prompts demand:
  `{"information_need": ...}` (reformulation) and a batched
  `{"verdicts": [...]}` (verification), with every `chunk_id` returned
  verbatim (`exact_match=True`) — `Verifier` can join verdicts to chunks
  without fuzzy matching.
- Recovery path: `model_b` at `max_new_tokens=200` produced `content=""` with
  `finish_reason="length"`; the one-shot rescue call (replaying the abandoned
  chain as an assistant turn) returned a parseable `ANSWER:` line.

## 3. `model_a` vs `model_b` probe (n=4, plumbing evidence, not an accuracy estimate)

Gold letter varied across the 4 closed-book questions.

| model | correct | wall time | completion tokens | notes |
| --- | --- | --- | --- | --- |
| `model_a` | 4/4 | 3.5–5.0 s | 217–379 | no `reasoning` field |
| `model_b` | 2/4 | 10.4–13.7 s | 801–1,058 | every call `finish_reason="stop"`; chains 2.9k–4.8k chars |

`model_b`'s two misses were substantive (Lambert-Eaton read as myasthenia
gravis; sarcoidosis as Lyme disease), not truncation — the 16,384-token
ceiling was never reached. Whether it can come down is a Phase 5 question
(needs a larger sample).

## 4. Thinking-mode probe (measured, not adopted)

`extra_body={"chat_template_kwargs": {"thinking": False}}` was accepted and
cut one identical request from 28.1 s / 465 completion tokens to 5.5 s / 231
tokens (~5x) while still emitting a parseable `ANSWER:` line, though it did
not eliminate the reasoning field entirely (2052 → 625 chars).

Not adopted for Phase 4: whether the model thinks is part of what the study
measures, so turning it off is a design lever, not a performance tweak —
decide at Phase 10 (open question 11).

## 5. Throughput caveat (feeds Phases 5/13)

Passing `seed` disables batched serving in `mlx_lm.server`
(`is_batchable and args.seed is None`), so requests serialize per endpoint.
Accepted for determinism, but `model_b` is one request at a time and will
dominate any full-run wall clock.

## 6. Decisions this produced

| Decision | Evidence | Where it lands |
| --- | --- | --- |
| `model_a`/`model_b` = Qwen2.5-7B-Instruct-8bit / DeepSeek-R1-Distill-Qwen-7B-8bit, on separate endpoints | §3 above | `config/models.yaml` |
| `max_new_tokens` 512 → 1024 | 512 truncated `model_b` before `ANSWER:` | `config/models.yaml` |
| `ModelConfig.timeout=120s`, `max_retries=4` | transport policy needs a config home, not hardcoding | `src/medical_rag/config.py` |
| `ModelConfig.answer_recovery_max_tokens` (0 = off) | §2 recovery-path verification | `src/medical_rag/config.py` |
| Thinking mode left on | §4 — design lever, not perf tweak | deferred to Phase 10 |

## 7. Open questions settled / moved

- §6 item 1 (real model endpoints): **settled** — see `environment.md`.
- §6 item 2 (what "two models" means): **settled** — instruct/reasoning pair
  configured and verified live (§3). Still open: whether instruct-vs-reasoning
  is the axis the study wants (Phase 10).
- §6 item 10 (does `seed` bite): **not yet settled** — deferred to Phase 5.
- §6 item 11 (thinking mode): **not yet settled** — deferred to Phase 10, this
  file carries the per-model latency/token numbers (§4).
