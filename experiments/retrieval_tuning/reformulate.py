"""Query reformulation, made inspectable and non-silent.

The first tuning run's `reform_dense` column is worthless, and the reason is
here rather than in retrieval: `judge_harness.reformulate_query()` did
`json.loads(result.content)` on a free-form completion, so anything that was not
*bare* JSON -- a ```json fence, a leading "Sure:", a reasoning model's preamble
-- threw, was caught by `except (JSONDecodeError, AttributeError)`, and returned
the **original question**. On 18 of 20 questions (measured:
`candidates["dense"] == candidates["reform_dense"]`) the "reformulation" lever
was a no-op, and the row still looked successful because the fallback was only a
`logger.warning` nobody re-read.

Two changes, both load-bearing:

1. **Parsing is tolerant** (fence-stripping + a targeted field regex, with
   `agenerate_structured` as a second attempt), so a well-formed intent is not
   lost to the punctuation wrapped around it.
2. **Failure is data.** The result carries `fallback` and `error`, the raw
   response and the prompt, and the inspector renders them. A fallback that is
   invisible is how a dead lever survives into a design freeze.

The prompt is overridable from a file so it can be tweaked without touching
production code -- `TUNING.md`'s rule that exploratory prompts live here, not in
`src/medical_rag/generation/prompt.py`, which Phase 8's `Verifier` and Phase 7's
`Reformulator` still depend on.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import openai
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.prompt import build_reformulation_prompt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_PROMPT_PATH = _HERE / "reform_prompt.txt"

REFORMULATION_SYSTEM_PROMPT: str = (
    "You rewrite medical exam questions into retrieval queries for a clinical "
    "reference database. You do not answer them."
)


class Reformulation(BaseModel):
    """The structured output the reformulation prompt asks for."""

    information_need: str = Field(min_length=1)


class ReformulationResult(BaseModel):
    """What the reformulator actually did, not just what it returned.

    `query` is always usable -- on failure it is the original question -- but
    `fallback` says whether that usability cost anything, and `error` says what.
    """

    query: str
    prompt: str
    raw_response: str | None = None
    fallback: bool = False
    error: str | None = None
    method: str = "structured"
    model: str | None = None
    prompt_source: str = "production build_reformulation_prompt()"


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
_FIELD_RE = re.compile(r'"information_need"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_information_need(raw: str) -> str | None:
    """Pull `information_need` out of a completion that may be fenced or padded.

    Deliberately not one `json.loads`: the bug this exists to fix was a correct
    answer wrapped in prose. Ordered strictest to loosest, and each step demands
    a non-empty string, so `{"information_need": ""}` still counts as a fallback
    rather than as a "rewrite" that retrieves the question minus one word.
    """
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    candidates = [fenced.group(1).strip(), text] if fenced else [text]

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            need = data.get("information_need")
            if isinstance(need, str) and need.strip():
                return need.strip()

    field = _FIELD_RE.search(text)
    if field:
        value = field.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if value:
            return value
    return None


def _is_transport(exc: BaseException) -> bool:
    """A dead or refusing endpoint, which must not be mistaken for bad output.

    `LLMError` wraps the real cause, and `EndpointCircuitBreaker` keys off the
    same `__cause__`, so both agree on what "the server is down" means.
    Swallowing a connection error into a fallback would report an outage as a
    result -- 20 questions quietly labelled "unreformulated".
    """
    return isinstance(
        exc.__cause__,
        (openai.APIConnectionError, openai.APIStatusError, openai.APITimeoutError),
    )



def load_template(path: str | Path | None = None) -> str | None:
    """Read a local prompt override, or None to use production's builder.

    `DEFAULT_PROMPT_PATH` is *not* loaded implicitly -- an experiment should not
    start diverging from production because someone left a file behind. Passing
    `--reform-prompt` is the opt-in, and the file it names is what `context.md`
    then points at.
    """
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"--reform-prompt {path} is empty")
    return text


def build_prompt(question: str, template: str | None = None) -> str:
    """Render the reformulation prompt -- production wording, or the override.

    A template without `{question}` gets the question appended rather than
    raising: the common tweak is to paste the production wording minus the JSON
    line back in, and silently emitting a prompt that never mentions the question
    would produce a fluent, useless rewrite of nothing.
    """
    if template is None:
        return build_reformulation_prompt(question)
    if "{question}" in template:
        return template.replace("{question}", question)
    return f"{template}\n\nQuestion: {question}"


async def reformulate_query(
    question: MedQAQuestion,
    client: LLMClient,
    *,
    template: str | None = None,
) -> ReformulationResult:
    """Rewrite `question` into an information need; never fail silently.

    Two attempts, then an explicit fallback to the question text. Transport
    errors re-raise so the caller's circuit breaker counts them.
    """
    prompt = build_prompt(question.question, template)
    source = (
        "production build_reformulation_prompt()"
        if template is None
        else "local --reform-prompt override"
    )
    model = client.config.name
    parse_error: str | None = None

    try:
        result = await client.agenerate(prompt, system_prompt=REFORMULATION_SYSTEM_PROMPT)
    except LLMError as exc:
        if _is_transport(exc):
            raise
        parse_error = f"agenerate: {type(exc).__name__}: {str(exc)[:180]}"
        logger.warning("q{}: reformulation call failed -- {}", question.id, parse_error)
    else:
        need = parse_information_need(result.content)
        if need:
            return ReformulationResult(
                query=need,
                prompt=prompt,
                raw_response=result.content,
                method="agenerate + tolerant parse",
                model=model,
                prompt_source=source,
            )
        parse_error = "no parsable information_need in the completion"

    try:
        structured = await client.agenerate_structured(
            prompt, Reformulation, system_prompt=REFORMULATION_SYSTEM_PROMPT
        )
    except (LLMError, ValidationError) as exc:
        if _is_transport(exc):
            raise
        return _fallback(
            question, prompt, source, model, f"{parse_error}; retry: {str(exc)[:180]}"
        )

    if structured.information_need.strip():
        return ReformulationResult(
            query=structured.information_need.strip(),
            prompt=prompt,
            raw_response=structured.model_dump_json(),
            method="agenerate_structured (retry)",
            model=model,
            prompt_source=source,
        )
    return _fallback(question, prompt, source, model, f"{parse_error}; retry returned an empty field")


def _fallback(
    question: MedQAQuestion, prompt: str, source: str, model: str, error: str
) -> ReformulationResult:
    logger.warning(
        "q{}: reformulation fell back to the raw question -- {} "
        "(the reform column will be identical to the orig column)",
        question.id,
        error,
    )
    return ReformulationResult(
        query=question.question,
        prompt=prompt,
        raw_response=None,
        fallback=True,
        error=error,
        method="fallback (original question)",
        model=model,
        prompt_source=source,
    )


def manual_result(query: str, *, model: str | None = None) -> ReformulationResult:
    """A hand-written information need -- tweak the lever without an LLM call.

    In the REPL this is `use <text>`: the cheapest way to watch `dense__orig`
    diverge from `dense__reform` without paying for a generation per guess.
    """
    text = query.strip()
    return ReformulationResult(
        query=text,
        prompt=f"(no prompt -- this query was typed by hand)\n{text}",
        raw_response=text,
        method="manual",
        model=model,
        prompt_source="manual --query / `use` override",
    )
