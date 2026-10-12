"""The retrieval-tuning question sample: build, load, and pin.

Stage B of R2 enlarges the judged sample from the pinned 20 to 100 questions,
by **containment**: the sample is the pinned 20 plus 80 hash-ranked dev ids,
and --sample-seed 2 is part of its definition. Nothing about the old sample
changes -- the same 20 ids appear first, in their original order, so an n=100
run *is* the old run on its first 20 questions and every cached verdict for
those carries over.

Why a file, not a flag: --question-ids with 100 typed ids is one fat-fingered
paste from silently running the wrong sample. The sample lives in a committed
JSON artifact with its seed, provenance and de-duplication rule recorded in
it, and `--check` refuses to serve a run built from a file that no longer
reproduces its own recorded construction. (Stage E2.)

Determinism: `hashlib.blake2b(id.encode(), digest_size=8)` -- NOT Python's
hash(), which is salted per process. Re-running the builder on the same pool
must reproduce byte-identical files; that is the whole content of "pinned".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.eval.runlog import DEFAULT_SAMPLE_SEED, select_by_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_PATH = REPO_ROOT / "eval" / "sample_v2.json"
LEGACY_SAMPLE_IDS: list[str] = [
    "1312", "2391", "3998", "4002", "5209", "5949", "6205", "6264", "6727",
    "6753", "6771", "7159", "7502", "7553", "8966", "9293", "9473", "9572",
    "9597", "10064",
]
SAMPLE_V2_SIZE = 100
SAMPLE_V2_SEED = 2
SAMPLE_V2_VERSION = "sample-v2-containment"


def stable_rank_key(seed: int, question_id: str) -> tuple[bytes, str]:
    """The deterministic rank key: blake2b of `seed:id`, then the id itself.

    The seed is mixed into the digest so it is part of the sample's definition
    (R2_plan §B: rank by `sha256(f"{seed}:{id}")` -- same idea, blake2b for the
    shorter fixed-size digest), not merely recorded next to it: seed 2 and seed
    3 must draw visibly different fills, or the seed is decoration. The id
    tiebreak makes collisions into a total order, so two builders can never
    disagree about a permutation even in principle. blake2b, not Python's
    hash(), which is salted per process.
    """
    return (
        hashlib.blake2b(f"{seed}:{question_id}".encode("utf-8"), digest_size=8).digest(),
        question_id,
    )


def _load_pool(split: str) -> list[MedQAQuestion]:
    pool = load_medqa(split)
    if len(pool) < SAMPLE_V2_SIZE:
        raise SystemExit(f"{split} has only {len(pool)} questions; need {SAMPLE_V2_SIZE}")
    return pool


def build_sample(
    split: str = "dev",
    size: int = SAMPLE_V2_SIZE,
    pinned: Sequence[str] = LEGACY_SAMPLE_IDS,
    seed: int = SAMPLE_V2_SEED,
) -> dict[str, Any]:
    """The containment sample: pinned ids first, hash-ranked fill to `size`.

    Returns the artifact dict; `write_sample` turns it into the JSON file. The
    construction is deliberately boring -- every id must resolve in the split
    (a typo in the pinned list is a build error, not a silent 99-question
    sample), pinned ids are de-duplicated preserving first occurrence, and the
    fill comes from the remaining pool in blake2b order. Seed is recorded so a
    future sample-v3 can change it *visibly*.
    """
    seen: set[str] = set()
    pinned_unique: list[str] = []
    for question_id in pinned:
        if question_id not in seen:
            seen.add(question_id)
            pinned_unique.append(question_id)
    if len(pinned_unique) != len(pinned):
        raise SystemExit("pinned ids contain duplicates after first-occurrence dedupe")

    pool = _load_pool(split)
    pool_by_id = {question.id: question for question in pool}
    missing = [question_id for question_id in pinned_unique if question_id not in pool_by_id]
    if missing:
        raise SystemExit(f"pinned id(s) not in {split} split: {', '.join(missing)}")

    if size < len(pinned_unique):
        raise SystemExit(f"size {size} is smaller than the {len(pinned_unique)} pinned ids")
    remaining = [question.id for question in pool if question.id not in seen]
    fill = sorted(remaining, key=lambda qid: stable_rank_key(seed, qid))[: size - len(pinned_unique)]

    ids = pinned_unique + fill
    selected = select_by_ids(pool, ids)
    gold = {question.id: question.answer_idx.upper() for question in selected}
    return {
        "version": SAMPLE_V2_VERSION,
        "seed": seed,
        "split": split,
        "size": len(ids),
        "pinned_ids": pinned_unique,
        "pinned_source": "harness.SAMPLE_QUESTION_IDS (R1's pinned 20, unmodified)",
        "fill_rule": f"blake2b(digest_size=8) of f'{{seed}}:{{question_id}}' ranked ascending "
        f"over remaining {split} ids (seed={seed}); id ascending breaks digest ties",
        "ids": ids,
        "gold_letters": gold,
    }


def write_sample(
    path: str | Path = DEFAULT_SAMPLE_PATH,
    split: str = "dev",
    size: int = SAMPLE_V2_SIZE,
    seed: int = SAMPLE_V2_SEED,
) -> dict[str, Any]:
    """Build and write the sample artifact; returns it for the caller to echo."""
    artifact = build_sample(split=split, size=size, seed=seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def read_sample(path: str | Path = DEFAULT_SAMPLE_PATH) -> dict[str, Any]:
    """Read a sample artifact and check the invariants a consumer relies on."""
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"{target} does not exist -- build it with `sample.py --build` first")
    artifact = json.loads(target.read_text(encoding="utf-8"))
    for key in ("version", "split", "size", "ids", "pinned_ids", "gold_letters"):
        if key not in artifact:
            raise SystemExit(f"{target} is missing '{key}' -- not a sample artifact")
    ids = artifact["ids"]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{target}: duplicate ids in 'ids'")
    if artifact["size"] != len(ids):
        raise SystemExit(f"{target}: size={artifact['size']} but {len(ids)} ids listed")
    if artifact["pinned_ids"] != ids[: len(artifact["pinned_ids"])]:
        raise SystemExit(f"{target}: pinned ids are not a prefix of 'ids' -- containment broken")
    return artifact


def check_sample(
    path: str | Path = DEFAULT_SAMPLE_PATH,
    split: str | None = None,
    seed: int | None = None,
    size: int | None = None,
) -> bool:
    """Rebuild from the artifact's recorded recipe and compare id-for-id (E2).

    The comparison is on the *id list and gold letters*, not on cosmetic dict
    order: a regeneration that produces the same ids but a changed
    `fill_rule` string is still a mismatch, because the file's claim about how
    it was built no longer holds.
    """
    artifact = read_sample(path)
    rebuilt = build_sample(
        split=split or artifact["split"],
        size=size or artifact["size"],
        pinned=artifact["pinned_ids"],
        seed=seed if seed is not None else artifact["seed"],
    )
    ok_ids = rebuilt["ids"] == artifact["ids"]
    ok_gold = rebuilt["gold_letters"] == artifact["gold_letters"]
    return ok_ids and ok_gold


def load_sample_file(path: str | Path) -> list[str]:
    """Ids for `--question-ids-file`: sample artifacts or a plain id list.

    Both shapes are accepted so a hand-written retry list works too -- one id
    per line, `#` comments, blank lines ignored.
    """
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"{target} does not exist")
    text = target.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("{"):
        artifact = json.loads(stripped)
        if "ids" not in artifact:
            raise SystemExit(f"{target}: JSON but no 'ids' key -- not a sample artifact")
        return list(artifact["ids"])
    ids = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    if not ids:
        raise SystemExit(f"{target}: no ids found")
    return ids


def sample_table(artifact: dict[str, Any]) -> str:
    """The pinned-vs-fill breakdown, for context.md and FINDINGS."""
    pinned = set(artifact["pinned_ids"])
    fill = [question_id for question_id in artifact["ids"] if question_id not in pinned]
    return (
        f"sample {artifact.get('version', '?')}: {artifact['size']} questions "
        f"= {len(pinned)} pinned + {len(fill)} hash-ranked "
        f"(split={artifact['split']}, seed={artifact.get('seed', '?')})"
    )


def render_sample_markdown(artifact: dict[str, Any]) -> str:
    """A findings-ready paragraph describing the sample and its construction."""
    pinned = set(artifact["pinned_ids"])
    fill = [question_id for question_id in artifact["ids"] if question_id not in pinned]
    gold = artifact.get("gold_letters") or {}
    tally: dict[str, int] = {}
    for letter in gold.values():
        tally[letter] = tally.get(letter, 0) + 1
    letters = ", ".join(f"{letter}={count}" for letter, count in sorted(tally.items())) or "-"
    return "\n".join(
        [
            "## Sample",
            "",
            f"{sample_table(artifact)}.",
            "",
            f"- pinned 20 (unchanged from the R1 run): {', '.join(artifact['pinned_ids'])}",
            f"- hash-ranked fill (first 10 of {len(fill)}): {', '.join(fill[:10])}"
            + (" ..." if len(fill) > 10 else ""),
            f"- gold letters across the sample: {letters}",
            "",
            "Containment means the n=100 run subsumes the n=20 run: its first 20 rows are "
            "the same questions in the same order, judged under the same rubric sha, so "
            "cached verdicts carry over and per-question comparisons stay valid.",
        ]
    )


@dataclass(frozen=True)
class SampleCheckResult:
    """What `--check` learned, for the exit line rather than a bare bool."""

    ok: bool
    path: Path
    detail: str


def verify_sample(path: str | Path = DEFAULT_SAMPLE_PATH) -> SampleCheckResult:
    """`--check` with a human-readable verdict string."""
    target = Path(path)
    try:
        ok = check_sample(target)
    except SystemExit as exc:
        return SampleCheckResult(False, target, str(exc))
    if ok:
        artifact = read_sample(target)
        return SampleCheckResult(True, target, sample_table(artifact))
    return SampleCheckResult(False, target, "rebuilt id list or gold letters differ from the file")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--build", action="store_true", help="build/write the sample artifact")
    parser.add_argument(
        "--check", action="store_true",
        help="rebuild from the recorded recipe and compare id-for-id (Stage E2 gate)",
    )
    parser.add_argument("--path", default=str(DEFAULT_SAMPLE_PATH), help="sample artifact path")
    parser.add_argument("--split", default=None, help="MedQA split (default: the file's, or dev)")
    parser.add_argument("--size", type=int, default=None, help="sample size (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="recorded rank seed (default: 2)")
    args = parser.parse_args(argv)

    if not args.build and not args.check:
        parser.print_help()
        return 2
    if args.check:
        result = verify_sample(args.path)
        print(f"{'OK' if result.ok else 'MISMATCH'}: {result.path} -- {result.detail}")
        return 0 if result.ok else 1
    artifact = write_sample(
        args.path,
        split=args.split or "dev",
        size=args.size or SAMPLE_V2_SIZE,
        seed=args.seed if args.seed is not None else SAMPLE_V2_SEED,
    )
    print(f"wrote {args.path}")
    print(sample_table(artifact))
    return 0


if __name__ == "__main__":
    sys.exit(main())