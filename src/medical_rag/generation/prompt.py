"""Prompt templates for answer generation.

Assembles a MedQA question, its answer options, and retrieved StatPearls
excerpts into the single string prompt `LLMClient.agenerate()` sends to the
remote endpoint, plus `parse_answer()` -- the parser for the answer format
these prompts mandate (the inverse of `ANSWER_FORMAT_INSTRUCTION`).

Prompt wording is provisional until the Phase 10 design freeze; the two
phrasings that measurably move results are `_RAG_CONTEXT_INSTRUCTION` (how
strongly the model is bound to the retrieved excerpts) and whether chain-of-
thought is requested.
"""

import re
from collections.abc import Sequence

from medical_rag.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT: str = (
    "You are a medical expert answering USMLE-style multiple-choice questions."
)

ANSWER_FORMAT_INSTRUCTION: str = (
    "Think step by step, then end your response with exactly one line in the "
    "form:\nANSWER: X\nwhere X is the letter of the single best option."
)

# Sent as a follow-up user turn when a completion produced no parsable ANSWER:
# line (see LLMClient._recover_answer). Worded to stop a reasoning model from
# restarting the chain it just abandoned, which is what burned the token budget.
ANSWER_RECOVERY_INSTRUCTION: str = (
    "Continue from where you stopped above. Do not restart or repeat your "
    "reasoning. Finish it briefly, then end your response with exactly one line "
    "in the form:\nANSWER: X\nwhere X is the letter of the single best option."
)

_CLOSED_BOOK_INSTRUCTION: str = "Answer from your own medical knowledge."

_RAG_CONTEXT_INSTRUCTION: str = (
    "Base your answer primarily on the clinical reference excerpts above. If "
    "they do not contain the answer, use your best clinical judgment."
)

_VERIFICATION_JSON_INSTRUCTION: str = (
    "Respond with JSON only, in exactly this shape:\n"
    '{"verdicts": [{"chunk_id": "<the chunk_id shown above>", "keep": true, '
    '"reason": "<one short sentence>"}]}\n'
    "Include exactly one verdict per chunk_id listed above, using each "
    'chunk_id verbatim. Set "keep" to true only if the excerpt helps answer '
    "the question or rules an option out."
)

_REFORMULATION_JSON_INSTRUCTION: str = (
    "Respond with JSON only, in exactly this shape:\n"
    '{"information_need": "<the clinical question that must be answered to '
    'solve the question above>"}\n'
    "Phrase it as a retrieval query for a medical reference: name the "
    "condition, finding, or mechanism being asked about, in your own words."
)


def format_options(options: dict[str, str]) -> str:
    """Render options letter-sorted, one per line, as `A. text`."""
    return "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks as numbered `[n] title` + content blocks."""
    return "\n\n".join(
        f"[{index}] {chunk.title}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )


def build_answer_prompt(
    question: str,
    options: dict[str, str],
    context_chunks: list[RetrievedChunk] | None,
) -> str:
    """Build the answer-generation prompt.

    `context_chunks=None` (or `[]`) renders the closed-book variant; a
    non-empty list renders the retrieval-augmented variant with the excerpts
    numbered in the order given.
    """
    sections = [f"Question: {question}", f"Options:\n{format_options(options)}"]

    if context_chunks:
        sections.append(f"Clinical reference excerpts:\n{format_context(context_chunks)}")
        sections.append(_RAG_CONTEXT_INSTRUCTION)
    else:
        sections.append(_CLOSED_BOOK_INSTRUCTION)

    sections.append(ANSWER_FORMAT_INSTRUCTION)
    return "\n\n".join(sections)


def build_reformulation_prompt(question: str) -> str:
    """Build the query-reformulation prompt (question only, by design).

    The answer options are deliberately withheld: the reformulated query is
    what retrieval is run on, and leaking an option's wording into it would
    let a condition retrieve the answer rather than the information need.
    """
    return "\n\n".join(
        [
            "Rewrite the following medical exam question as a single, concise "
            "information-need statement suitable for searching a clinical "
            "reference database. Do not answer it.\n\n"
            f"Question: {question}",
            _REFORMULATION_JSON_INSTRUCTION,
        ]
    )


def build_verification_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build one batched chunk-relevance prompt for all candidate chunks.

    All chunks go into a single prompt (one LLM call per question, not one
    per chunk) and each block echoes its `chunk_id` verbatim, so
    `Verifier.verify()` can join verdicts back to chunks exactly.
    """
    numbered = "\n\n".join(
        f"[{index}] chunk_id={chunk.chunk_id} title={chunk.title}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return "\n\n".join(
        [
            "You are screening retrieved reference excerpts for relevance to a "
            "medical exam question.\n\n"
            f"Question: {question}",
            f"Candidate excerpts:\n{numbered}",
            _VERIFICATION_JSON_INSTRUCTION,
        ]
    )


# Preference order for parse_answer: an explicit `ANSWER: X` marker wins over
# `the answer is X` prose, which wins over a bare letter alone on the last
# line. Within a tier the *last* match wins, so a decoy marker quoted earlier
# in a chain-of-thought response cannot beat the model's final answer.
_ANSWER_MARKER_RE = re.compile(
    r"(?<![A-Za-z])answer\b[^A-Za-z0-9\n]{0,8}\(?[*_]{0,2}([A-Za-z])(?![A-Za-z])",
    re.IGNORECASE,
)
_ANSWER_IS_RE = re.compile(
    r"(?<![A-Za-z])answer\s+(?:is|would\s+be|should\s+be|will\s+be)\s*"
    r"\(?[*_]{0,2}([A-Za-z])(?![A-Za-z])",
    re.IGNORECASE,
)
_BARE_LETTER_RE = re.compile(
    r"(?<![A-Za-z])\(?[*_]{0,2}([A-Za-z])[*_]{0,2}\)?(?![A-Za-z])"
)
_MAX_FINAL_LINE_CHARS = 40


def _candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for regex in (_ANSWER_MARKER_RE, _ANSWER_IS_RE):
        matches = [match.group(1).upper() for match in regex.finditer(text)]
        candidates.extend(reversed(matches))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= _MAX_FINAL_LINE_CHARS:
        bare = [match.group(1).upper() for match in _BARE_LETTER_RE.finditer(lines[-1])]
        candidates.extend(reversed(bare))

    return candidates


def parse_answer(text: str, valid_letters: set[str] | None = None) -> str | None:
    """Extract the chosen option letter from a model response, or None.

    Returns an upper-case letter. When `valid_letters` is given (e.g. the keys
    of `MedQAQuestion.options`), candidates outside that set are rejected
    rather than returned. `None` means nothing parseable was found; callers
    log `parsed_answer=None` / `is_correct=False` rather than guessing.
    """
    if not text:
        return None

    allowed = {letter.upper() for letter in valid_letters} if valid_letters else set("ABCDE")
    for candidate in _candidates(text):
        if candidate in allowed:
            return candidate

    return None

