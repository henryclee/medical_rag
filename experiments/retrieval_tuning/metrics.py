"""Recall/relevance formulas, shared by `FINDINGS.md` and the inspection grid.

The point of one implementation: when a cell in the browser shows `recall@5`
and the aggregate table shows a different number for the same method, neither
is believable. `judge_harness.compute_metrics()` delegates here.

Every formula takes the gold letter *as data* -- the caller reads it off the
dataset, the judge never sees it (`judge_prompt.py`).
"""

from __future__ import annotations

from typing import Any, Sequence

K_VALUES: tuple[int, ...] = (5, 10, 20)
RELEVANT_LEVELS = {"relevant", "partial"}

# A question with no relevant chunk in the window is scored at this rank rather
# than dropped, so `mean_first_relevant_rank` cannot improve by a method
# silently surfacing fewer chunks.
BEYOND_K = 99


def gold_hit(chunk_ids: Sequence[str], judgments: dict[str, Any], gold: str) -> bool:
    """Did any chunk in this window claim to give evidence for the gold option?"""
    return any(gold in (judgments.get(c) or {}).get("supports_options", []) for c in chunk_ids)


def relevant_fraction(chunk_ids: Sequence[str], judgments: dict[str, Any]) -> float:
    """Share of the window judged `relevant` or `partial`. Empty window -> 0.0."""
    if not chunk_ids:
        return 0.0
    hits = sum(1 for c in chunk_ids if (judgments.get(c) or {}).get("relevance") in RELEVANT_LEVELS)
    return hits / len(chunk_ids)


def first_relevant_rank(chunk_ids: Sequence[str], judgments: dict[str, Any]) -> int | None:
    """1-based rank of the first `relevant` chunk, or None when there is none.

    `partial` deliberately does not count: "topically related but does not
    resolve the question" is what `relevant_fraction@k` already measures.
    Folding it in here would make the one number that matters -- how deep you
    must read to find something usable -- optimistic.
    """
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if (judgments.get(chunk_id) or {}).get("relevance") == "relevant":
            return rank
    return None


def cell_metrics(
    chunk_ids: Sequence[str],
    judgments: dict[str, Any],
    gold: str,
    ks: Sequence[int] = K_VALUES,
) -> dict[str, Any]:
    """The single-question row the inspector prints under each cell."""
    return {
        "n_chunks": len(chunk_ids),
        "per_k": {
            k: {
                "recall": gold_hit(chunk_ids[:k], judgments, gold),
                "relevant_fraction": round(relevant_fraction(chunk_ids[:k], judgments), 3),
            }
            for k in ks
        },
        "first_relevant_rank": first_relevant_rank(chunk_ids, judgments),
        "gold_positions": [
            rank
            for rank, chunk_id in enumerate(chunk_ids, start=1)
            if gold in (judgments.get(chunk_id) or {}).get("supports_options", [])
        ],
    }


def aggregate(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    ks: Sequence[int] = K_VALUES,
) -> dict[str, Any]:
    """Per-method means over judged rows -- the shape `FINDINGS.md` renders.

    Reads only successful rows. A method a row does not carry (e.g. the reform
    column when `--no-reform` ran) lowers that method's `n` rather than being
    counted as a miss, because a missing column is not a measured zero.
    """
    usable = [row for row in rows if "error" not in row]
    summary: dict[str, Any] = {"n_questions": len(usable), "methods": {}}
    if not usable:
        return summary

    for method in methods:
        present = [row for row in usable if row.get("candidates", {}).get(method) is not None]
        n = len(present)
        per_k = {k: {"recall_hits": 0, "fractions": []} for k in ks}
        first_ranks: list[int] = []
        for row in present:
            candidates = row["candidates"][method]
            judgments = row.get("judgments") or {}
            gold = row["correct_answer"]
            for k in ks:
                window = candidates[:k]
                per_k[k]["recall_hits"] += int(gold_hit(window, judgments, gold))
                per_k[k]["fractions"].append(relevant_fraction(window, judgments))
            rank = first_relevant_rank(candidates, judgments)
            if rank is not None:
                first_ranks.append(rank)

        summary["methods"][method] = {
            "n": n,
            "per_k": {
                k: {
                    "semantic_recall": round(per_k[k]["recall_hits"] / n, 3) if n else None,
                    "relevant_fraction": round(sum(per_k[k]["fractions"]) / n, 3) if n else None,
                }
                for k in ks
            },
            "mean_first_relevant_rank": round(sum(first_ranks) / len(first_ranks), 2)
            if first_ranks
            else None,
            "n_first_relevant": len(first_ranks),
        }
    return summary
