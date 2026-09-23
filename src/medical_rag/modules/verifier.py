"""Answer verification module.

Will check a generated answer against retrieved context using the remote
LLM, for experiment conditions that use answer verification/self-checking.

TODO: class Verifier: wraps an LLM client (see generation.llm).
TODO: Verifier.verify(question, answer, context) -> verification result.
"""
