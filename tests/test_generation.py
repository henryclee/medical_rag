"""Tests for the prompt builders, the answer parser, and the remote LLM client.

Prompt and parser tests are pure. Every `LLMClient` test runs against
`httpx.MockTransport`, so it exercises the real request/response path --
headers, JSON body, retries, error translation -- with no network. One test
hits the real endpoint and is gated on `MEDICAL_RAG_LIVE_TESTS=1`.
"""

import json
import os

import httpx
import pytest
from loguru import logger
from pydantic import BaseModel

from medical_rag.config import ModelConfig, load_config
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.prompt import (
    ANSWER_FORMAT_INSTRUCTION,
    ANSWER_RECOVERY_INSTRUCTION,
    build_answer_prompt,
    build_reformulation_prompt,
    build_verification_prompt,
    format_context,
    format_options,
    parse_answer,
)
from medical_rag.retrieval.retriever import RetrievedChunk

OPTIONS = {
    "A": "Hyperkalemia",
    "B": "Hypokalemia",
    "C": "Hypercalcemia",
    "D": "Hyponatremia",
}


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"statpearls_doc_{index}::paragraph_1",
        content=f"Body text of chunk {index}.",
        title=f"StatPearls Title {index}",
        score=0.9 - 0.1 * index,
    )


# --- prompt builders ---------------------------------------------------------


def test_build_answer_prompt_contains_question_and_every_option():
    prompt = build_answer_prompt("Which electrolyte abnormality is this?", OPTIONS, None)

    assert "Which electrolyte abnormality is this?" in prompt
    for letter, option_text in OPTIONS.items():
        assert f"\n{letter}. {option_text}" in prompt
    assert prompt.endswith(ANSWER_FORMAT_INSTRUCTION)


@pytest.mark.parametrize("context_chunks", [None, []])
def test_closed_book_variant_has_no_context_section(context_chunks):
    prompt = build_answer_prompt("Question?", OPTIONS, context_chunks)

    assert "Clinical reference excerpts" not in prompt
    assert "Answer from your own medical knowledge." in prompt


def test_rag_variant_numbers_excerpts_and_keeps_titles():
    chunks = [_chunk(1), _chunk(2)]
    prompt = build_answer_prompt("Question?", OPTIONS, chunks)

    assert "[1] StatPearls Title 1" in prompt
    assert "[2] StatPearls Title 2" in prompt
    assert "Body text of chunk 2." in prompt
    # the adherence instruction distinguishes RAG from closed-book
    assert "Base your answer primarily on the clinical reference excerpts above" in prompt
    assert "Answer from your own medical knowledge." not in prompt
    assert prompt.index("[1]") < prompt.index("[2]")


def test_format_helpers_render_options_and_context():
    assert format_options({"B": "Beta", "A": "Alpha"}) == "A. Alpha\nB. Beta"
    assert format_context([_chunk(1)]) == (
        "[1] StatPearls Title 1\nBody text of chunk 1."
    )
    assert format_context([]) == ""


def test_reformulation_prompt_withholds_the_options():
    prompt = build_reformulation_prompt("What mechanism explains the ECG findings?")

    assert "What mechanism explains the ECG findings?" in prompt
    assert "information_need" in prompt
    assert "Do not answer it" in prompt
    for option_text in OPTIONS.values():
        assert option_text not in prompt


def test_verification_prompt_echoes_each_chunk_id_exactly_once():
    chunks = [_chunk(index) for index in range(1, 6)]
    prompt = build_verification_prompt("Question?", chunks)

    for chunk in chunks:
        assert prompt.count(chunk.chunk_id) == 1
    assert "[5] chunk_id=statpearls_doc_5::paragraph_1" in prompt
    assert '"keep"' in prompt


# --- parse_answer ------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning about the case.\nANSWER: C", "C"),
        ("reasoning\nanswer: c", "C"),
        ("**ANSWER: (B)**", "B"),
        ("ANSWER: D is correct because sodium is low.", "D"),
        ("I first wrote ANSWER: A, but on reflection:\nANSWER: B", "B"),
        ("The answer is E.", "E"),
        ("Weighing the potassium level against the ECG.\nB", "B"),
        ("Final answer would be (D), given the history.", "D"),
        ("I cannot determine the answer.", None),
        ("", None),
    ],
)
def test_parse_answer(response, expected):
    assert parse_answer(response) == expected


def test_parse_answer_rejects_letters_outside_the_option_set():
    assert parse_answer("ANSWER: E", valid_letters=set(OPTIONS)) is None
    assert parse_answer("ANSWER: b", valid_letters={"A", "B"}) == "B"


# --- LLMClient (offline, against httpx.MockTransport) ------------------------

API_KEY_ENV = "TEST_LLM_API_KEY"


def _config(**overrides) -> ModelConfig:
    payload = {
        "name": "test_model",
        "base_url": "http://testserver/v1",
        "api_model_name": "served-model",
        "api_key_env": API_KEY_ENV,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 0,
    }
    payload.update(overrides)
    return ModelConfig(**payload)


def _completion(
    content="ANSWER: B",
    finish_reason="stop",
    reasoning=None,
    reasoning_field="reasoning",
) -> dict:
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        # A field the OpenAI SDK does not know about, mirroring what
        # mlx_lm.server (`reasoning`) and oMLX (`reasoning_content`) send.
        message[reasoning_field] = reasoning
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "served-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _client(handler, **config_overrides) -> LLMClient:
    transport = httpx.MockTransport(handler)
    return LLMClient(
        _config(**config_overrides),
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "test-key-123")


@pytest.fixture
def logged_warnings():
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(sink_id)


def test_missing_api_key_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(LLMError, match=API_KEY_ENV):
        LLMClient(_config())


def test_transport_policy_comes_from_config(api_key):
    client = LLMClient(
        _config(timeout=3.5, max_retries=2),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_completion()))),
    )

    assert client.client.max_retries == 2
    assert client.client.timeout == 3.5


async def test_agenerate_returns_stripped_content(api_key):
    def handler(request):
        return httpx.Response(200, json=_completion("  ANSWER: B  "))

    async with _client(handler) as client:
        assert (await client.agenerate("Question?")).content == "ANSWER: B"


async def test_agenerate_sends_config_defaults_and_credentials(api_key):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion())

    async with _client(handler) as client:
        await client.agenerate("Question?")

    body = seen["body"]
    assert seen["auth"] == "Bearer test-key-123"
    assert body["model"] == "served-model"
    assert body["max_completion_tokens"] == 512
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["seed"] == 0
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][-1]["content"] == "Question?"


async def test_per_call_kwargs_override_config_values(api_key):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion())

    async with _client(handler) as client:
        await client.agenerate("Question?", seed=7, max_new_tokens=64, temperature=0.0, top_p=0.5)

    body = seen["body"]
    assert body["seed"] == 7
    assert body["max_completion_tokens"] == 64
    assert body["temperature"] == 0.0  # 0.0 must not read as "use the config value"
    assert body["top_p"] == 0.5


async def test_system_prompt_can_be_suppressed(api_key):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion())

    async with _client(handler) as client:
        await client.agenerate("Question?", system_prompt=None)

    assert [message["role"] for message in seen["body"]["messages"]] == ["user"]


async def test_transient_rate_limit_is_retried(api_key):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(
                429, json={"error": {"message": "slow down", "type": "rate_limit_error"}}
            )
        return httpx.Response(200, json=_completion())

    async with _client(handler, max_retries=2) as client:
        assert (await client.agenerate("Question?")).content == "ANSWER: B"

    assert len(calls) == 2


async def test_persistent_server_error_becomes_llmerror(api_key):
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}})

    async with _client(handler, max_retries=0) as client:
        with pytest.raises(LLMError, match="HTTP 500"):
            await client.agenerate("Question?")


async def test_auth_failure_names_the_env_var(api_key):
    def handler(request):
        return httpx.Response(
            401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
        )

    async with _client(handler, max_retries=0) as client:
        with pytest.raises(LLMError, match=API_KEY_ENV):
            await client.agenerate("Question?")


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [(None, "length"), ("", "stop"), ("   ", "stop")],
)
async def test_unusable_content_raises(api_key, content, finish_reason):
    def handler(request):
        return httpx.Response(200, json=_completion(content, finish_reason))

    async with _client(handler) as client:
        with pytest.raises(LLMError, match="empty content"):
            await client.agenerate("Question?")


async def test_truncated_response_warns_but_still_returns(api_key, logged_warnings):
    def handler(request):
        return httpx.Response(200, json=_completion("A long-winded explanation", "length"))

    async with _client(handler) as client:
        assert (await client.agenerate("Question?")).content == "A long-winded explanation"

    assert any("finish_reason='length'" in message for message in logged_warnings)


# --- reasoning chain handling ------------------------------------------------


async def test_reasoning_is_kept_out_of_the_answer(api_key):
    """Scoring reads `content`; the chain of thought only rides along for logs."""

    def handler(request):
        return httpx.Response(
            200,
            json=_completion("ANSWER: B", reasoning="The ACE level points somewhere. Hmm."),
        )

    async with _client(handler) as client:
        result = await client.agenerate("Question?")

    assert result.content == "ANSWER: B"
    assert result.reasoning == "The ACE level points somewhere. Hmm."
    assert result.parsed_answer == "B"
    assert result.unanswered is False


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_both_chain_of_thought_field_names_are_read(api_key, field):
    def handler(request):
        return httpx.Response(
            200, json=_completion("", "length", reasoning="thinking", reasoning_field=field)
        )

    async with _client(handler) as client:
        result = await client.agenerate("Question?")

    assert result.reasoning == "thinking"


async def test_reasoning_only_completion_is_recorded_not_raised(api_key, logged_warnings):
    """Raising here used to lose the row; it must survive to be scored incorrect."""

    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_completion("", "length", reasoning="A very long chain."))

    async with _client(handler) as client:
        result = await client.agenerate("Question?")

    assert len(calls) == 1  # recovery is off unless answer_recovery_max_tokens is set
    assert result.content == ""
    assert result.reasoning == "A very long chain."
    assert result.truncated is True
    assert result.unanswered is True
    assert any("finish_reason='length'" in message for message in logged_warnings)


# --- answer recovery ---------------------------------------------------------


async def test_answered_completion_issues_no_recovery_call(api_key):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_completion())

    async with _client(handler, answer_recovery_max_tokens=256) as client:
        result = await client.agenerate("Question?")

    assert len(calls) == 1
    assert (result.recovery_attempted, result.recovered) == (False, False)


async def test_unanswered_completion_gets_one_recovery_call(api_key):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(
                200, json=_completion("", "length", reasoning="Weighing A against C.")
            )
        return httpx.Response(200, json=_completion("ANSWER: C"))

    async with _client(handler, answer_recovery_max_tokens=256) as client:
        result = await client.agenerate("Question?")

    assert (result.recovery_attempted, result.recovered) == (True, True)
    assert result.parsed_answer == "C"
    assert result.unanswered is False
    assert len(bodies) == 2
    # The rescue continues the abandoned chain rather than starting a fresh one,
    # and stays inside its own small budget.
    replayed = [m for m in bodies[1]["messages"] if m["role"] == "assistant"]
    assert replayed[0]["content"] == "Weighing A against C."
    assert bodies[1]["messages"][-1]["content"] == ANSWER_RECOVERY_INSTRUCTION
    assert bodies[1]["max_completion_tokens"] == 256
    # Cost accounting spans both calls.
    assert result.completion_tokens == 14


async def test_failed_recovery_leaves_the_row_unanswered(api_key, logged_warnings):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(
            200, json=_completion("still thinking", "length", reasoning="More thinking.")
        )

    async with _client(handler, answer_recovery_max_tokens=256) as client:
        result = await client.agenerate("Question?")

    assert len(calls) == 2  # one rescue attempt, never a loop
    assert result.recovery_attempted is True
    assert result.recovered is False
    assert result.unanswered is True
    assert result.content == "still thinking"  # the original text is not discarded
    assert any("no parsable ANSWER: line after a recovery attempt" in m for m in logged_warnings)


# --- structured output -------------------------------------------------------


class InformationNeed(BaseModel):
    information_need: str


async def test_agenerate_structured_validates_the_payload(api_key):
    def handler(request):
        return httpx.Response(
            200, json=_completion('{"information_need": "cause of hypercalcemia"}')
        )

    async with _client(handler) as client:
        output = await client.agenerate_structured("Question?", InformationNeed)

    assert output == InformationNeed(information_need="cause of hypercalcemia")


async def test_agenerate_structured_wraps_unparseable_output(api_key):
    def handler(request):
        return httpx.Response(200, json=_completion("I am not JSON at all"))

    async with _client(handler) as client:
        with pytest.raises(LLMError, match="structured generation"):
            await client.agenerate_structured("Question?", InformationNeed)


# --- live smoke check --------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("MEDICAL_RAG_LIVE_TESTS") != "1",
    reason="set MEDICAL_RAG_LIVE_TESTS=1 to call the real endpoint",
)
@pytest.mark.parametrize("model_key", ["model_a", "model_b"])
async def test_live_endpoint_answers_a_medqa_style_question(model_key):
    model = load_config().models[model_key]
    options = {
        "A": "Sarcoidosis",
        "B": "Lyme disease",
        "C": "Guillain-Barre syndrome",
        "D": "Multiple sclerosis",
    }
    prompt = build_answer_prompt(
        "A 45-year-old woman presents with bilateral facial nerve palsy, uveitis, and "
        "elevated serum ACE. Which of the following is the most likely diagnosis?",
        options,
        None,
    )

    async with LLMClient(model) as client:
        result = await client.agenerate(prompt, seed=model.seed)

    # model_b thinks in a `reasoning` field and can spend its whole budget doing
    # it, so require that *something* came back, and score only `content`.
    assert result.content or result.reasoning
    assert result.parsed_answer in set(options)


