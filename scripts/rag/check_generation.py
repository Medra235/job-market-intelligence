"""Run a manual generation check against a local Ollama model.

Usage examples:
    .venv/bin/python scripts/rag/check_generation.py --prompt "Give me three Data Engineering skills."
    .venv/bin/python scripts/rag/check_generation.py --prompt "Summarize MLOps in one paragraph." --num-predict 256
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config.settings import settings  # noqa: E402
from src.rag.generator import (  # noqa: E402
    GenerationConfig,
    generate_answer,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local generation testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt to send to the local Ollama model.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for the model.",
    )
    parser.add_argument(
        "--model-name",
        default=settings.ollama_model,
        help="Ollama model name to use.",
    )
    parser.add_argument(
        "--ollama-url",
        default=settings.ollama_url,
        help="Ollama base URL.",
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
        "--num-predict",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable separate thinking output for models that support it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one local generation request."""
    args = parse_args()
    config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.num_predict,
        think=args.think,
    )
    answer = generate_answer(
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        model_name=args.model_name,
        base_url=args.ollama_url,
        config=config,
    )
    print(answer)


if __name__ == "__main__":
    main()
