"""Recompute the load-bearing claims of an exploration run from its artifacts.

Phase 5's postmortem found that a number can be reported from a contaminated run
and look perfectly ordinary, and Phase 6 nearly repeated it with a `tok_s` that
was really queue latency. A test suite cannot catch either: the code was correct
and the *conditions* were wrong. So the audit reads `results.jsonl`/`pairing.md`
back and asserts the identities the report leans on -- the pair counts, the
arm-to-arm context parity, the accuracy reconciliation, and whether any row was
pinned at its request ceiling (which would make a latency claim a capacity claim).

Not a test: nothing here fails the build. It prints PASS/FAIL lines to be read
before a number leaves the repo.

    .venv/bin/python scripts/exploration/audit_run.py outputs/exploration/phase6/<run>
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "outputs/exploration/phase6/20260928T143811Z"
)
rows = [json.loads(line) for line in (RUN / "results.jsonl").read_text().splitlines() if line]
by_key = {(r["model"], r["question_id"]): r for r in rows}
models = sorted({r["model"] for r in rows})
questions = sorted({r["question_id"] for r in rows})

print(f"rows={len(rows)} models={models} questions={len(questions)}")
print(f"artifacts={[p.name for p in sorted(RUN.iterdir())]}")


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label} {detail}")


med = statistics.median


check("row count = arms x questions", len(rows) == len(models) * len(questions))
check("no duplicate keys", len(by_key) == len(rows))
check("condition_id on every row", all(r.get("condition_id") for r in rows))
check("retrieved_chunk_ids on every row", all(r.get("retrieved_chunk_ids") for r in rows))

outcomes = Counter(r["outcome"] for r in rows)
check("outcome on every row", all(r.get("outcome") for r in rows), str(dict(outcomes)))

base_right = sum(1 for r in rows if r["base_correct"])
rag_right = sum(1 for r in rows if r["is_correct"])
# `distracted` + `lost` are the rows RAG cost; `rescued` is the set it gained.
# `both_wrong`/`both_right` are invariant, so they cancel -- counting them as
# "broke" (an earlier draft of this script did) makes the identity pass by
# construction and prints a meaningless "fixed" number.
cost = outcomes["distracted"] + outcomes["lost"]
gained = outcomes["rescued"]
check("closed -> rag arithmetic reconciles", base_right - cost + gained == rag_right,
      f"(base right {base_right}, rag right {rag_right}, gained {gained}, cost {cost})")
check("rag-wrong rows = lost + distracted + both_wrong",
      len(rows) - rag_right == outcomes["lost"] + outcomes["distracted"] + outcomes["both_wrong"])
check("no unanswered rows", not any(r["unanswered"] for r in rows))
check("no error rows", not any("error" in r for r in rows))

problems = []
for qid in questions:
    group = [by_key[(m, qid)] for m in models if (m, qid) in by_key]
    if len(group) != len(models):
        problems.append(f"q{qid}: only {len(group)}/{len(models)} arms")
        continue
    # `base_answer` is deliberately NOT compared: it is each arm's own closed-book
    # answer, so the two arms legitimately differ. The parity that must hold is the
    # retrieved context -- the two arms answer the *same* prompt.
    if len({tuple(r["retrieved_chunk_ids"]) for r in group}) != 1:
        problems.append(f"q{qid}: chunk ids differ")
check("both arms saw identical context per question", not problems, str(problems[:3]))

timings = {}
for r in rows:  # both arms share one retrieval pass, so time each question once
    timings[r["question_id"]] = r["retrieval"]["retrieve_s"] + r["retrieval"]["rerank_s"]
vals = sorted(timings.values())
print(f"retrieval+rerank per question (one shared pass, not per row): median "
      f"{statistics.median(vals):.2f}s max {max(vals):.2f}s sum {sum(vals):.1f}s")
print(f"  of which vector search: median "
      f"{statistics.median([r['retrieval']['retrieve_s'] for r in rows]):.3f}s | reranker: median "
      f"{statistics.median([r['retrieval']['rerank_s'] for r in rows]):.3f}s "
      f"max {max(r['retrieval']['rerank_s'] for r in rows):.3f}s (cold start)")
for m in models:
    mine = [r for r in rows if r["model"] == m]
    walls = [r["wall_s"] for r in mine]
    tps = [r["tok_s"] for r in mine if r["completion_tokens"] >= 64]
    ptoks = [r["prompt_tokens"] for r in mine]
    print(f"  {m}: wall median {statistics.median(walls):.2f}s max {max(walls):.2f}s | "
          f"tok/s median {statistics.median(tps):.1f} (n={len(tps)}) | "
          f"prompt_tokens {min(ptoks)}-{max(ptoks)}")
    check(f"{m}: nothing pinned at the ceiling",
          not any(r["finish_reason"] == "length" for r in mine),
          f"length-finishes={sum(1 for r in mine if r['finish_reason'] == 'length')}")

print("\ndistractors (baseline right -> RAG wrong):")
for r in rows:
    if r["outcome"] != "distracted":
        continue
    print(f"  q{r['question_id']} {r['model']} base={r['base_answer']} rag={r['parsed_answer']} "
          f"gold_in_context={r['gold_chunk_rank']} answer_in_context={r['answer_chunk_rank']} "
          f"rerank_top={r['retrieval']['rerank_top_score']:.2f} "
          f"vector_top1={r['retrieval']['vector_top1_score']:.3f}")

print("\nrescued (baseline wrong -> RAG right):")
for r in rows:
    if r["outcome"] != "rescued":
        continue
    print(f"  q{r['question_id']} {r['model']} base={r['base_answer']} rag={r['parsed_answer']} "
          f"gold_rank={r['gold_chunk_rank']} answer_rank={r['answer_chunk_rank']}")

absent = [r for r in rows if r["parsed_answer"] and r["answer_chunk_rank"] is None]
print(f"\nrows whose chosen answer text is absent from the context: {len(absent)}/{len(rows)} "
      f"({len(absent) / len(rows):.0%})")
changed = [r for r in rows if r["answer_changed"]]
check("answer_changed implies a baseline pair", all(r["base_answer"] is not None for r in changed),
      f"{len(changed)} changed")

ranked = defaultdict(int)
for r in rows:
    if r["gold_chunk_rank"]:
        ranked[r["gold_chunk_rank"]] += 1
print(f"gold chunk rank histogram: {dict(sorted(ranked.items()))}")
chunks = [c for r in rows for c in r["chunks"]]
promoted = sum(1 for c in chunks if c["vector_rank"] and c["prompt_rank"] < c["vector_rank"])
print(f"rerank promotions (prompt_rank < vector_rank): {promoted}/{len(chunks)} chunks")
print(f"vector scores collapsed: {sum(1 for r in rows if r['retrieval']['vector_top1_score'] - r['retrieval']['vector_last_score'] < 0.05)} rows")

# The cost of context, read from the baseline run *pairing.md* names rather than
# from a hard-coded path: a latency claim that silently compares two different
# endpoints or two different concurrency settings is worse than no claim.
_header = (RUN / "pairing.md").read_text().splitlines()[0]
_found = re.search(r"baseline ([\w/-]+)", _header)
_baseline_dir = Path("outputs/exploration") / _found.group(1) if _found else None
if _baseline_dir and _baseline_dir.is_dir():
    _brows = [json.loads(l) for l in (_baseline_dir / "results.jsonl").read_text().splitlines() if l]
    _byb = {(r["model"], r["question_id"]): r for r in _brows}
    print(f"\ncost of context -- closed-book ({_baseline_dir.name}) -> RAG, same arm, same question:")
    for m in models:
        pairs = [(_byb[(r["model"], r["question_id"])], r) for r in rows
                 if r["model"] == m and (r["model"], r["question_id"]) in _byb]
        if not pairs:
            continue
        # `reasoning_chars` is 0 by construction for a model that emits no thinking
        # block (model_a), so say so only when the whole arm is flat-zero.
        nothink = all(not b["reasoning_chars"] and not r["reasoning_chars"] for b, r in pairs)
        print(f"  {m}: n={len(pairs)} | completion tokens median "
              f"{med([b['completion_tokens'] for b, _ in pairs]):.0f} -> "
              f"{med([r['completion_tokens'] for _, r in pairs]):.0f} | median wall "
              f"{med([b['wall_s'] for b, _ in pairs]):.1f}s -> "
              f"{med([r['wall_s'] for _, r in pairs]):.1f}s | reasoning chars median "
              f"{med([b['reasoning_chars'] for b, _ in pairs]):.0f} -> "
              f"{med([r['reasoning_chars'] for _, r in pairs]):.0f}"
              + (" (this model emits no thinking block)" if nothink else ""))
        # Delta medians are per-question paired differences, which is what "cost of
        # context" means here; they are NOT the difference of the two medians above,
        # and their spread (chunk length) is wider than the shift suggests.
        delta = [r["prompt_tokens"] - b["prompt_tokens"] for b, r in pairs]
        print(f"    prompt tokens median {med([b['prompt_tokens'] for b, _ in pairs]):.0f} -> "
              f"{med([r['prompt_tokens'] for _, r in pairs]):.0f} | paired delta from the "
              f"excerpts: median {med(delta):.0f} (min {min(delta)}, max {max(delta)}) | "
              f"finish_reason length: "
              f"{sum(1 for _, r in pairs if r['finish_reason'] == 'length')} of {len(pairs)}")
else:
    print(f"\n[WARN] baseline rows for the closed-book-vs-RAG cost not found: {_baseline_dir}")

print("\nper-model transitions (closed-book -> RAG on the same question):")
for m in models:
    mine = [r for r in rows if r["model"] == m]
    counts = Counter(r["outcome"] for r in mine)
    base = sum(1 for r in mine if r["base_correct"])
    rag = sum(1 for r in mine if r["is_correct"])
    print(f"  {m}: closed {base}/{len(mine)} -> rag {rag}/{len(mine)} | rescued "
          f"{counts['rescued']} distracted {counts['distracted']} lost {counts['lost']} "
          f"both_right {counts['both_right']} both_wrong {counts['both_wrong']}")
    errs = Counter(r["parsed_answer"] or "?" for r in mine if not r["is_correct"])
    print(f"    error letters: {dict(sorted(errs.items()))} | recovery fired: "
          f"{sum(1 for r in mine if r['recovery_attempted'])}")
    print(f"    tokens: prompt {sum(r['prompt_tokens'] for r in mine)}, completion "
          f"{sum(r['completion_tokens'] for r in mine)}, generation wall "
          f"{sum(r['wall_s'] for r in mine):.0f}s")

print("\n--- pairing.md ---")
print((RUN / "pairing.md").read_text())
