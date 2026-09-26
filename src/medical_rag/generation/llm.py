"""Remote LLM client.

Wraps an OpenAI-compatible chat-completions endpoint (`openai.AsyncOpenAI`)
pointed at a separately served model described by a `ModelConfig`. No model
weights are loaded in-process -- every generation is an HTTP call.

`agenerate()` returns a `GenerationResult`: the answer text plus the chain of
thought a reasoning model returns separately. mlx_lm.server puts that chain in
a non-standard `reasoning` field (oMLX-style servers call it
`reasoning_content`), so both names are read off the response message's extra
fields. `agenerate_structured()` returns a validated pydantic model through
pydantic-ai using *prompted* JSON output rather than tool calling: neither
model served here honoured `tool_choice="required"` when probed in Phase 4 --
both returned JSON text in `content` with no tool call. Native
`response_format` JSON mode does work against these endpoints and is the
documented fallback if prompted output ever misbehaves.

A completion that never reaches a parsable `ANSWER:` line -- a reasoning model
that spent its whole budget thinking -- is rescued once via
`ModelConfig.answer_recovery_max_tokens`. A row that still fails is returned as
`GenerationResult.unanswered` for the runner to record and score incorrect,
rather than raising away.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeVar

import httpx
import openai
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.openai import OpenAIProvider

from medical_rag.config import ModelConfig
from medical_rag.generation.prompt import (
    ANSWER_RECOVERY_INSTRUCTION,
    SYSTEM_PROMPT,
    parse_answer,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """A generation call failed, or returned nothing usable."""


# mlx_lm.server labels the chain of thought `reasoning`; oMLX-style servers use
# `reasoning_content`. Both are checked, in this order.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass(slots=True)
class GenerationResult:
    """One completion, with a reasoning model's chain of thought kept separate.

    `content` is what scoring, citation matching, and the Judge must use.
    `reasoning` is kept for logs and qualitative analysis only: R1-style chains
    restate retrieved passages verbatim, so scoring them would inflate every
    citation-support and support-rate measure.
    """

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_s: float = 0.0
    recovery_attempted: bool = False
    recovery_content: str = ""
    recovered: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def parsed_answer(self) -> str | None:
        """The option letter that will be scored, or None."""
        return parse_answer(self.content)

    @property
    def unanswered(self) -> bool:
        """No parsable answer after any recovery attempt -- scored incorrect."""
        return self.parsed_answer is None


def _extract_reasoning(message: object) -> str:
    """Read the chain of thought out of a completion message, whatever it's named.

    The OpenAI SDK keeps server-added fields in `model_extra` and mirrors them as
    attributes, so either route is accepted and an endpoint that renames the field
    still works.
    """
    extras = getattr(message, "model_extra", None) or {}
    for field in _REASONING_FIELDS:
        value = extras.get(field) if isinstance(extras, dict) else None
        if value is None:
            value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class LLMClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build a client for one served model.

        `http_client` is a test seam: pass an `httpx.AsyncClient` built on
        `httpx.MockTransport` to exercise request/response behaviour offline.
        """
        self.config = config

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise LLMError(
                f"model '{config.name}' reads its API key from environment variable "
                f"'{config.api_key_env}', which is unset or empty; export it or add it "
                "to the repo's gitignored .env (see .env.example)"
            )

        self.client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _resolve_params(
        self,
        seed: int | None,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> dict:
        config = self.config
        return {
            "seed": config.seed if seed is None else seed,
            "max_new_tokens": config.max_new_tokens if max_new_tokens is None else max_new_tokens,
            "temperature": config.temperature if temperature is None else temperature,
            "top_p": config.top_p if top_p is None else top_p,
        }

    async def agenerate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = SYSTEM_PROMPT,
    ) -> GenerationResult:
        """Generate once; per-call kwargs override the ModelConfig defaults.

        `LLMError` is raised only when the call yielded nothing at all. A
        reasoning-only or truncated completion is returned instead -- after one
        rescue attempt through `answer_recovery_max_tokens` -- so the runner can
        record the row and score it as unanswered rather than losing it.
        """
        config = self.config
        params = self._resolve_params(seed, max_new_tokens, temperature, top_p)

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stamp = datetime.now(timezone.utc).isoformat()
        logger.bind(kind="llm_request", model=config.name, seed=params["seed"], timestamp=stamp).debug(
            "prompt: {}", prompt
        )

        result = await self._complete(messages, **params)

        logger.bind(kind="llm_summary", model=config.name, seed=params["seed"], timestamp=stamp).info(
            "llm call model={} seed={} finish_reason={} elapsed={:.2f}s "
            "prompt_tokens={} completion_tokens={} response_chars={} reasoning_chars={}",
            config.name,
            params["seed"],
            result.finish_reason,
            result.elapsed_s,
            result.prompt_tokens,
            result.completion_tokens,
            len(result.content),
            len(result.reasoning),
        )
        logger.bind(kind="llm_response", model=config.name, seed=params["seed"], timestamp=stamp).debug(
            "response: {}", result.content
        )
        if result.reasoning:
            logger.bind(
                kind="llm_reasoning", model=config.name, seed=params["seed"], timestamp=stamp
            ).debug("chain of thought: {}", result.reasoning)

        if result.truncated:
            logger.warning(
                "model '{}' exhausted max_new_tokens={} before finishing "
                "(finish_reason='length'; {} reasoning chars, {} answer chars) -- "
                "the ANSWER: line may have been truncated away",
                config.name,
                params["max_new_tokens"],
                len(result.reasoning),
                len(result.content),
            )
        if not result.content and not result.reasoning:
            raise LLMError(
                f"model '{config.name}' returned empty content "
                f"(finish_reason={result.finish_reason!r}) with no reasoning to "
                "recover an answer from"
            )

        if result.unanswered and config.answer_recovery_max_tokens > 0:
            result = await self._recover_answer(messages, result, params)

        if result.unanswered:
            logger.warning(
                "model '{}' produced no parsable ANSWER: line{}; recording this "
                "generation as unanswered and scoring it incorrect",
                config.name,
                " after a recovery attempt" if result.recovery_attempted else "",
            )
        return result

    async def _complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> GenerationResult:
        """Send one chat completion, splitting answer text from chain of thought."""
        config = self.config
        started = time.perf_counter()
        try:
            completion = await self.client.chat.completions.create(
                model=config.api_model_name,
                messages=messages,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        except openai.AuthenticationError as exc:
            raise LLMError(
                f"endpoint {config.base_url} rejected the key for model '{config.name}' "
                f"(HTTP 401); check {config.api_key_env} in .env"
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMError(
                f"endpoint {config.base_url} returned HTTP {exc.status_code} for model "
                f"'{config.name}' after {config.max_retries} retries: {str(exc)[:200]}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"could not reach {config.base_url} for model '{config.name}': {exc}"
            ) from exc
        elapsed = time.perf_counter() - started

        if not completion.choices:
            raise LLMError(f"model '{config.name}' returned no choices")

        choice = completion.choices[0]
        usage = completion.usage
        return GenerationResult(
            content=(choice.message.content or "").strip(),
            reasoning=_extract_reasoning(choice.message),
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            elapsed_s=elapsed,
        )

    async def _recover_answer(
        self,
        messages: list[dict],
        result: GenerationResult,
        params: dict,
    ) -> GenerationResult:
        """Give a completion that never stated an answer one chance to close.

        Everything the model produced is replayed as an assistant turn so the
        rescue continues the chain it abandoned instead of starting a new one --
        restarting is what burned the previous budget. The recovered text becomes
        `content`; the chain of thought is kept.
        """
        config = self.config
        produced = "\n\n".join(part for part in (result.reasoning, result.content) if part.strip())
        logger.warning(
            "model '{}' spent {} tokens without a parsable ANSWER: line "
            "(finish_reason={!r}, {} reasoning chars) -- issuing one recovery "
            "call of {} tokens",
            config.name,
            result.completion_tokens,
            result.finish_reason,
            len(result.reasoning),
            config.answer_recovery_max_tokens,
        )

        recovery = await self._complete(
            [
                *messages,
                {"role": "assistant", "content": produced},
                {"role": "user", "content": ANSWER_RECOVERY_INSTRUCTION},
            ],
            max_new_tokens=config.answer_recovery_max_tokens,
            temperature=params["temperature"],
            top_p=params["top_p"],
            seed=params["seed"],
        )

        result.recovery_attempted = True
        result.recovery_content = recovery.content
        result.recovered = recovery.parsed_answer is not None
        result.completion_tokens = (result.completion_tokens or 0) + (
            recovery.completion_tokens or 0
        )
        result.elapsed_s += recovery.elapsed_s
        if result.recovered:
            result.content = recovery.content
            if recovery.reasoning:
                result.reasoning += f"\n\n[recovery]\n{recovery.reasoning}"
            logger.info(
                "model '{}' recovered ANSWER={} in {:.2f}s on a {} token recovery call",
                config.name,
                recovery.parsed_answer,
                recovery.elapsed_s,
                config.answer_recovery_max_tokens,
            )
        return result

    async def agenerate_structured(
        self,
        prompt: str,
        output_type: type[ModelT],
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = SYSTEM_PROMPT,
    ) -> ModelT:
        """Generate once and validate the response against `output_type`."""
        config = self.config
        params = self._resolve_params(seed, max_new_tokens, temperature, top_p)

        logger.bind(kind="llm_request", model=config.name, structured=True).debug(
            "structured prompt ({}): {}", output_type.__name__, prompt
        )

        agent = Agent(
            OpenAIChatModel(
                config.api_model_name,
                provider=OpenAIProvider(openai_client=self.client),
            ),
            output_type=PromptedOutput(
                output_type,
                name=output_type.__name__,
                description=f"A validated {output_type.__name__}.",
            ),
            instructions=system_prompt,
            model_settings=OpenAIChatModelSettings(
                max_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                seed=params["seed"],
                timeout=config.timeout,
            ),
        )

        started = time.perf_counter()
        try:
            result = await agent.run(prompt)
        except Exception as exc:
            raise LLMError(
                f"structured generation for {output_type.__name__} failed against model "
                f"'{config.name}': {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc

        output = result.output
        logger.bind(kind="llm_summary", model=config.name, structured=True).info(
            "llm structured call model={} output={} elapsed={:.2f}s",
            config.name,
            output_type.__name__,
            time.perf_counter() - started,
        )
        logger.bind(kind="llm_response", model=config.name, structured=True).debug(
            "structured response: {}", output
        )

        return output if isinstance(output, output_type) else output_type.model_validate(output)

