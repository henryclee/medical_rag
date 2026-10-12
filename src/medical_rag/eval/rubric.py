"""The judge rubric: schema, prompt, and the hash the verdict cache is keyed on.

Lives here rather than in `eval/harness.py` because `eval/inspector.py`
issues the same call over the same chunks, and a viewer that re-grades with
drifted wording produces verdicts that look interchangeable with the harness's
but are not. Split out of the harness, wording unchanged, so the ~820 verdicts
the 20-question run already paid for stay valid under `JUDGE_PROMPT_VERSION`.

The judge is never shown the gold letter -- it reports which option(s) a chunk
gives evidence for, and gold recall is derived programmatically afterwards
(see `metrics.py`). That is what makes one verdict reusable across every
strategy instead of being re-graded per condition.

R2 (ADR-0008's gate) adds `RubricSpec`: the (version, system prompt, JSON
instruction) triple a cache sha is derived from, as a value instead of three
module globals. `DEFAULT_RUBRIC` carries the exact historical wording -- a test
asserts its sha stays `21e27467277a`, so the refactor invalidates nothing -- and
the R2 stability variant lives beside it under its own sha. `ensure_verdicts`
takes a `RubricSpec`, so a variant's verdicts land in the shared cache keyed by
*their* sha and `load_cache` filters by sha: two wordings can share one file
without either polluting the other. What was removed is the ability to stamp an
arbitrary `judge_sha` over unchanged wording -- that sha would have lied about
what produced the records.
"""

from __future__ import annotations

import hashlib
import dataclasses
from typing import Literal, Sequence

from pydantic import BaseModel

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.generation.prompt import format_options
from medical_rag.retrieval.retriever import RetrievedChunk

# Bump whenever the wording below changes. It is a version, not a date to read:
# the cache key is the rubric sha, so an edit here silently invalidates every
# cached verdict -- which is the point, but it is also why bumping it casually
# costs a full re-judge (~45-60 min of endpoint time for 20 questions).
JUDGE_PROMPT_VERSION = "2026-09-29.1"

RELEVANT_LEVELS = {"relevant", "partial"}


class ChunkJudgment(BaseModel):
    chunk_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    supports_options: list[str]
    reason: str


class JudgeVerdict(BaseModel):
    judgments: list[ChunkJudgment]


@dataclasses.dataclass(frozen=True)
class RubricSpec:
    """The exact triple a verdict-cache sha is derived from.

    A rubric is version + system prompt + JSON instruction, hashed together --
    nothing else. Question text and options are already in the cache key
    (`question_id`), and `chunk_id` identifies the chunk's content for a frozen
    index, so hashing the rendered prompt would key verdicts to their own inputs
    and never reuse. What must invalidate a cached verdict is a changed rubric --
    only that. Frozen and hashable so it can be a default argument and a cache
    key component without anyone mutating a wording a verdict already cites.
    """

    version: str
    system_prompt: str
    json_instruction: str

    def sha(self) -> str:
        material = "\n".join((self.version, self.system_prompt, self.json_instruction))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


DEFAULT_SYSTEM_PROMPT: str = (
    "You are grading whether retrieved medical reference excerpts help answer "
    "a USMLE-style multiple-choice question. You are not answering it."
)

DEFAULT_JSON_INSTRUCTION: str = (
    "Respond with JSON only, in exactly this shape:\n"
    '{"judgments": [{"chunk_id": "<the chunk_id shown above>", '
    '"relevance": "relevant" | "partial" | "irrelevant", '
    '"supports_options": ["<subset of the option letters above, or []>"], '
    '"reason": "<one short sentence>"}]}\n'
    "Include exactly one judgment per chunk_id listed above, using each "
    "chunk_id verbatim. \"relevant\" means the excerpt directly helps "
    "distinguish which option is correct; \"partial\" means it is topically "
    "related but does not resolve the question; \"irrelevant\" means it does "
    "not help at all. List an option letter in supports_options only if the "
    "excerpt gives evidence for that specific option -- never guess which "
    "option is correct if the excerpt does not say."
)

# The rubric the 926 verdicts of the 2026-09-29 runs were graded under. The
# wording is frozen: a test asserts this object's sha stays `21e27467277a`,
# so a rename or a re-flow of this module cannot quietly re-key the cache.
DEFAULT_RUBRIC = RubricSpec(
    version=JUDGE_PROMPT_VERSION,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    json_instruction=DEFAULT_JSON_INSTRUCTION,
)

# The R2 stability variant (r2-stability-a): identical shape, tightened on the
# one failure mode the hand-check controls probe -- case narrative /
# administrative / reporting boilerplate earning non-`irrelevant` verdicts
# because it mentions option-related terms. Its verdicts live in the same
# cache keyed by their own sha; nothing reads them unless the R2 gate adopts
# the variant, and adopting it is exactly what makes this the default (one
# edit, new sha, old verdicts stale-but-kept by design).
STABILITY_VARIANT_RUBRIC = RubricSpec(
    version="r2-stability-a",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    json_instruction=(
        "Respond with JSON only, in exactly this shape:\n"
        '{"judgments": [{"chunk_id": "<the chunk_id shown above>", '
        '"relevance": "relevant" | "partial" | "irrelevant", '
        '"supports_options": ["<subset of the option letters above, or []>"], '
        '"reason": "<one short sentence>"}]}\n'
        "Include exactly one judgment per chunk_id listed above, using each "
        "chunk_id verbatim. \"relevant\" requires a finding, quantity, "
        "mechanism, or guideline statement that discriminates between the "
        "options; case narrative, administrative, or reporting boilerplate is "
        "irrelevant even when it mentions option-related terms. \"partial\" "
        "means it is topically related but does not resolve the question; "
        "\"irrelevant\" means it does not help at all. List an option letter "
        "in supports_options only if the excerpt gives evidence for that "
        "specific option -- never guess which option is correct if the excerpt "
        "does not say. In the reason, name the sentence that discriminates."
    ),
)

# Backwards-compatible aliases: the cache, the harness, the inspector and the
# tests read these names, and they are exactly the default rubric's strings --
# never a second wording that could drift from the sha it is keyed under.
JUDGE_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
JUDGE_JSON_INSTRUCTION = DEFAULT_JSON_INSTRUCTION


def judge_prompt_sha() -> str:
    """Hash of the *default* rubric, not of a rendered prompt.

    Question text and options are already in the cache key (`question_id`), and
    `chunk_id` identifies the chunk's content for a frozen index, so hashing the
    rendered prompt would key verdicts to their own inputs and never reuse.
    What must invalidate a cached verdict is a changed rubric -- only that.
    Kept as a module function because the cache, the harness and the inspector
    all key on "the rubric in force" and there is exactly one: `DEFAULT_RUBRIC`.
    A variant under consideration is graded through `RubricSpec.sha()`
    explicitly (`ensure_verdicts(..., rubric=...)`); it never silently becomes
    the sha this returns.
    """
    return DEFAULT_RUBRIC.sha()


def build_judge_prompt(question: MedQAQuestion, chunks: Sequence[RetrievedChunk]) -> str:
    """One batched, gold-blind relevance prompt over every candidate chunk."""
    numbered = "\n\n".join(
        f"[{index}] chunk_id={chunk.chunk_id} title={chunk.title}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return "\n\n".join(
        [
            "You are grading retrieved reference excerpts for a medical exam "
            "question. You are NOT told which option is correct -- judge each "
            "excerpt on its own medical content.\n\n"
            f"Question: {question.question}",
            f"Options:\n{format_options(question.options)}",
            f"Candidate excerpts:\n{numbered}",
            DEFAULT_JSON_INSTRUCTION,
        ]
    )
