"""Endpoint preflight: is the served model the model `models.yaml` claims?

Moved out of `scripts/exploration/raw_rag.py`, which used to be the only home
for it and which the retrieval harness imported through -- so a Phase 6
answer-accuracy script was the dependency root for Phase-agnostic endpoint
hygiene. It is not measurement and it is not config, so it lives beside the
client it guards, and every harness that spends completions imports the same
check: the frozen Phase 5/6 answer runs and the judge harness alike.
"""

from enum import Enum

from medical_rag.config import ModelConfig
from medical_rag.generation.llm import LLMClient


class Preflight(Enum):
    """Whether an arm may spend its completions on the endpoint it was given."""

    PROCEED = "proceed"
    BLOCK = "block"


async def preflight(model: ModelConfig, client: LLMClient) -> tuple[Preflight, list[str]]:
    """Confirm the endpoint answers and serves the model `models.yaml` names.

    Returns a decision and the lines to print, rather than logging or raising: this
    runs once per arm before ~20 completions each, and the two failure modes want
    different handling. A dead endpoint blocks the whole run, but a misnamed
    `api_model_name` only invalidates *that* arm -- dropping the arm and pairing
    the other is a smaller loss than a run whose rows came from some other model.
    Phase 5 §8.4 is the reason `BLOCK` refuses rather than warns: an endpoint
    serving a different model produces rows that look identical and measure
    something else.

    Same reasoning makes it load-bearing for the judge under ADR-0008: if
    `:8080` has been repointed at another model, the verdict cache silently
    changes oracle mid-run and every cached verdict keeps looking like it came
    from the model `models.yaml` names.
    """
    try:
        served = await client.client.models.list()
    except Exception as exc:  # noqa: BLE001 - any failure here means "do not send"
        return Preflight.BLOCK, [
            f"{model.name}: endpoint {model.base_url} is not answering -- "
            f"{type(exc).__name__}: {str(exc)[:150]}. Start the server that serves this model "
            "(environment.md) or repoint config/models.yaml."
        ]

    ids = sorted({item.id for item in served.data if item.id})
    if model.api_model_name not in ids:
        return Preflight.BLOCK, [
            f"{model.name}: {model.base_url} serves {ids} but models.yaml names "
            f"'{model.api_model_name}' -- every completion would be rejected or, worse, "
            "answered by a different model than this arm claims"
        ]
    return Preflight.PROCEED, [
        f"{model.name}: {model.base_url} serves {model.api_model_name} (offering {', '.join(ids)})"
    ]
