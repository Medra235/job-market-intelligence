"""Run a manual generation check against a local LLM backend.

Usage examples:
    uv run python scripts/rag/check_generation.py --prompt "Give me three Data Engineering skills."
    uv run python scripts/rag/check_generation.py --prompt "Summarize MLOps in one paragraph." --max-tokens 256
"""

import argparse

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.generator import (
    GenerationConfig,
    LlamaCppGenerator,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local generation testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt to send to the local LLM backend.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for the model.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="llama.cpp model name to use.",
    )
    parser.add_argument(
        "--llamacpp-url",
        default=settings.llamacpp_url,
        help="llama.cpp server base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one local generation request."""
    args = parse_args()
    model_name = args.model_name or settings.llamacpp_model
    config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    generator = LlamaCppGenerator(model_name=model_name, base_url=args.llamacpp_url)
    answer = generator.generate(
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        config=config,
    )
    print(answer)


if __name__ == "__main__":
    main()
