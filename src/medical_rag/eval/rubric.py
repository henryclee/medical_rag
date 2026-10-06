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
"""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.generation.prompt import format_options
from medical_rag.retrieval.retriever import RetrievedChunk

# Bump whenever the wording below changes. It is a version, not a date to read:
# the cache key is `judge_prompt_sha()`, so an edit here silently invalidates
# every cached verdict -- which is the point, but it is also why bumping it
# casually costs a full re-judge (~45-60 min of endpoint time for 20 questions).
JUDGE_PROMPT_VERSION = "2026-09-29.1"

RELEVANT_LEVELS = {"relevant", "partial"}


class ChunkJudgment(BaseModel):
    chunk_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    supports_options: list[str]
    reason: str


class JudgeVerdict(BaseModel):
    judgments: list[ChunkJudgment]


JUDGE_SYSTEM_PROMPT: str = (
    "You are grading whether retrieved medical reference excerpts help answer "
    "a USMLE-style multiple-choice question. You are not answering it."
)

JUDGE_JSON_INSTRUCTION: str = (
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


def judge_prompt_sha() -> str:
    """Hash of the *rubric*, not of a rendered prompt.

    Question text and options are already in the cache key (`question_id`), and
    `chunk_id` identifies the chunk's content for a frozen index, so hashing the
    rendered prompt would key verdicts to their own inputs and never reuse.
    What must invalidate a cached verdict is a changed rubric -- only that.
    """
    material = "\n".join((JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, JUDGE_JSON_INSTRUCTION))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


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
            JUDGE_JSON_INSTRUCTION,
        ]
    )
