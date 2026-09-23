"""Remote LLM client.

Will wrap an OpenAI-compatible client (via the `openai` package and/or
pydantic-ai) pointed at a separately-served endpoint. No model weights are
loaded in-process.

TODO: class LLMClient: configured from models.yaml (base_url, model name, etc).
TODO: LLMClient.generate(prompt, **kwargs) -> str response.
TODO: LLMClient.agenerate(prompt, **kwargs) -> async str response.
"""
