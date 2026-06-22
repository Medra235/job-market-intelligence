"""Local LLM generation utilities backed by llama.cpp."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from job_market_intelligence.config.settings import settings


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings shared by local LLM backends."""

    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    extra_options: dict[str, Any] = field(default_factory=dict)

    def to_llamacpp_options(self) -> dict[str, Any]:
        """Return options using llama.cpp's OpenAI-compatible request keys."""
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            **self.extra_options,
        }
        chat_template_kwargs = dict(options.get("chat_template_kwargs", {}))
        chat_template_kwargs["enable_thinking"] = False
        options["chat_template_kwargs"] = chat_template_kwargs
        return options


class TextGenerator(Protocol):
    """Interface implemented by local LLM text generation adapters."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response for one prompt."""


class LlamaCppGenerator:
    """Small llama.cpp server client using the OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name: str = settings.llamacpp_model,
        base_url: str = settings.llamacpp_url,
        timeout_seconds: float = settings.llamacpp_timeout_seconds,
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
        """Generate a response for one prompt using llama.cpp's chat endpoint."""
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
            **request_config.to_llamacpp_options(),
        }

        try:
            response = self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not generate a response with llama.cpp. "
                "Check that llama.cpp server is running and the GGUF model is loaded."
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")
        if not content:
            raise RuntimeError("llama.cpp returned an empty response.")
        return content.strip()


def generate_answer(
    prompt: str,
    system_prompt: str | None = None,
    model_name: str = settings.llamacpp_model,
    base_url: str = settings.llamacpp_url,
    config: GenerationConfig | None = None,
) -> str:
    """Convenience function for one-off llama.cpp generation calls."""
    generator = LlamaCppGenerator(model_name=model_name, base_url=base_url)
    return generator.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
    )
