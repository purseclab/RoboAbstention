#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from parser import ParseError, parse_and_validate
from prompt import build_prompt


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = WORKSPACE_ROOT / ".env"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "openai/gpt-5.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 visual grounding.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"VLM provider identifier. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit LiteLLM model string. Overrides the provider default.",
    )
    return parser.parse_args()


def validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type {image_path.suffix!r}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def resolve_model(provider: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported provider {provider!r}. Supported providers: {sorted(DEFAULT_MODELS)}"
        )
    return DEFAULT_MODELS[provider]


def write_failure(output_path: Path, raw_response: str | None, errors: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failure",
        "raw_response": raw_response,
        "errors": errors,
    }
    output_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


@retry(
    wait=wait_random_exponential(multiplier=1, min=8, max=128),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_vlm(model: str, prompt: str, image_data_url: str) -> str:
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0,
        max_tokens=10000,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    try:
        validate_image_path(args.image)
        model = resolve_model(args.provider, args.model)
        prompt = build_prompt()
        image_data_url = image_to_data_url(args.image)
    except Exception as exc:  # noqa: BLE001
        write_failure(args.output, None, [str(exc)])
        print(str(exc), file=sys.stderr)
        return 1

    try:
        raw_response = call_vlm(model, prompt, image_data_url)
        validated = parse_and_validate(raw_response)
    except ParseError as exc:
        write_failure(args.output, exc.raw_response, exc.validation_errors)
        for error in exc.validation_errors:
            print(error, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        write_failure(args.output, None, [str(exc)])
        print(str(exc), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validated, indent=4) + "\n", encoding="utf-8")
    print(
        f"Phase 1 complete: {len(validated['scene_objects'])} objects, "
        f"{len(validated['scene_locations'])} locations -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
