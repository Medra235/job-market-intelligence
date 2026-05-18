"""Local LLM generation utilities backed by Ollama."""

from dataclasses import dataclass, field
from typing import Any

import httpx

from job_market_intelligence.config.settings import settings


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings sent to the local Ollama model."""

    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 512
    think: bool | str | None = False
    extra_options: dict[str, Any] = field(default_factory=dict)

    def to_ollama_options(self) -> dict[str, Any]:
        """Return options using Ollama's expected request keys."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            **self.extra_options,
        }


class OllamaGenerator:
    """Small Ollama client for single-turn text generation."""

    def __init__(
        self,
        model_name: str = settings.ollama_model,
        base_url: str = settings.ollama_url,
        timeout_seconds: float = settings.ollama_timeout_seconds,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response for one prompt using Ollama's chat endpoint."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        request_config = config or GenerationConfig()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": request_config.to_ollama_options(),
        }
        if request_config.think is not None:
            payload["think"] = request_config.think

        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not generate a response with Ollama. "
                "Check that Ollama is running and the model is available locally."
            ) from exc

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            if message.get("thinking"):
                raise RuntimeError(
                    "Ollama returned thinking output but no final answer. "
                    "Try disabling thinking for this model."
                )
            raise RuntimeError("Ollama returned an empty response.")
        return content.strip()


def generate_answer(
    prompt: str,
    system_prompt: str | None = None,
    model_name: str = settings.ollama_model,
    base_url: str = settings.ollama_url,
    config: GenerationConfig | None = None,
) -> str:
    """Convenience function for one-off Ollama generation calls."""
    generator = OllamaGenerator(model_name=model_name, base_url=base_url)
    return generator.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
    )
