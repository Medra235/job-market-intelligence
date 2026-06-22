"""Tests for the llama.cpp text generator."""

import json

import httpx
import pytest

from job_market_intelligence.rag.generator import GenerationConfig, LlamaCppGenerator


def test_generation_config_converts_to_llamacpp_options() -> None:
    """Generation settings should use llama.cpp's OpenAI-compatible names."""
    config = GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        max_tokens=256,
        extra_options={"min_p": 0.05},
    )

    assert config.to_llamacpp_options() == {
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 256,
        "min_p": 0.05,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_llamacpp_generator_sends_expected_chat_payload() -> None:
    """The generator should call llama.cpp's OpenAI-compatible chat endpoint."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(request.read().decode())
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Usa SQL, Python y Airflow.",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = LlamaCppGenerator(
        model_name="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
        base_url="http://llamacpp.test",
        client=client,
    )

    answer = generator.generate(
        prompt="Recommend skills for Data Engineering.",
        system_prompt="Answer in Spanish.",
        config=GenerationConfig(max_tokens=128),
    )

    assert answer == "Usa SQL, Python y Airflow."
    assert captured_request["url"] == "http://llamacpp.test/v1/chat/completions"
    assert captured_request["payload"] == {
        "model": "unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
        "messages": [
            {"role": "system", "content": "Answer in Spanish."},
            {
                "role": "user",
                "content": "Recommend skills for Data Engineering.",
            },
        ],
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize("prompt", ["", "   "])
def test_llamacpp_generator_rejects_empty_prompt(prompt: str) -> None:
    """Empty prompts should fail before calling llama.cpp."""
    generator = LlamaCppGenerator(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: None))
    )

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generator.generate(prompt=prompt)


def test_llamacpp_generator_wraps_http_errors() -> None:
    """HTTP failures should become a project-level runtime error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, json={"error": "model failed"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = LlamaCppGenerator(client=client)

    with pytest.raises(RuntimeError, match="llama.cpp"):
        generator.generate(prompt="Hello")


def test_llamacpp_generator_reports_empty_response() -> None:
    """Malformed or empty llama.cpp responses should fail explicitly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": ""}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = LlamaCppGenerator(client=client)

    with pytest.raises(RuntimeError, match="empty response"):
        generator.generate(prompt="Hello")
