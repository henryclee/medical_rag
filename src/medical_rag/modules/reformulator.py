"""Query reformulation module.

Will rewrite a raw question into one or more retrieval-optimized queries
using the remote LLM, for experiment conditions that use query reformulation.

TODO: class Reformulator: wraps an LLM client (see generation.llm).
TODO: Reformulator.reformulate(question) -> list[str] reformulated queries.
"""
