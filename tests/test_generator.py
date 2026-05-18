"""Tests for the Ollama-backed text generator."""

import httpx
import pytest

from job_market_intelligence.rag.generator import GenerationConfig, OllamaGenerator


def test_generation_config_converts_to_ollama_options() -> None:
    """Generation settings should use Ollama's option names."""
    config = GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        num_predict=256,
        think=False,
        extra_options={"num_ctx": 4096},
    )

    assert config.to_ollama_options() == {
        "temperature": 0.1,
        "top_p": 0.8,
        "num_predict": 256,
        "num_ctx": 4096,
    }


def test_ollama_generator_sends_expected_chat_payload() -> None:
    """The generator should call Ollama's chat endpoint with one user prompt."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = request.read().decode()
        return httpx.Response(
            status_code=200,
            json={"message": {"role": "assistant", "content": "Use SQL and Airflow."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OllamaGenerator(
        model_name="qwen3.5:9b",
        base_url="http://ollama.test",
        client=client,
    )

    answer = generator.generate(
        prompt="Recommend skills for Data Engineering.",
        system_prompt="Answer briefly.",
        config=GenerationConfig(num_predict=128),
    )

    assert answer == "Use SQL and Airflow."
    assert captured_request["url"] == "http://ollama.test/api/chat"
    assert '"model":"qwen3.5:9b"' in captured_request["payload"]
    assert '"role":"system"' in captured_request["payload"]
    assert '"role":"user"' in captured_request["payload"]
    assert '"stream":false' in captured_request["payload"]
    assert '"think":false' in captured_request["payload"]
    assert '"num_predict":128' in captured_request["payload"]


def test_ollama_generator_allows_thinking_when_requested() -> None:
    """Thinking can be enabled explicitly for models that support it."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["payload"] = request.read().decode()
        return httpx.Response(
            status_code=200,
            json={"message": {"role": "assistant", "content": "Final answer."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OllamaGenerator(client=client)

    answer = generator.generate(
        prompt="Hello",
        config=GenerationConfig(think=True),
    )

    assert answer == "Final answer."
    assert '"think":true' in captured_request["payload"]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_ollama_generator_rejects_empty_prompt(prompt: str) -> None:
    """Empty prompts should fail before calling Ollama."""
    generator = OllamaGenerator(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: None))
    )

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generator.generate(prompt=prompt)


def test_ollama_generator_wraps_http_errors() -> None:
    """HTTP failures should become a project-level runtime error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, json={"error": "model failed"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OllamaGenerator(client=client)

    with pytest.raises(RuntimeError, match="Could not generate"):
        generator.generate(prompt="Hello")


def test_ollama_generator_reports_thinking_without_final_answer() -> None:
    """Thinking-only responses should fail with a helpful error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "Internal reasoning only.",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = OllamaGenerator(client=client)

    with pytest.raises(RuntimeError, match="thinking output"):
        generator.generate(prompt="Hello")
